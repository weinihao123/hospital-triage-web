'use strict';
/**
 * 浏览器端分诊引擎（纯前端，无后端依赖）。
 * 精确移植自服务端 core/ 逻辑：
 *   tokenizer.createSegmenter + normalize
 *   indexer.retrieverQuery / retriever.retrieve（MiniSearch 反序列化 + 科室信号聚合）
 *   scorer.score（五路信号 softmax 归一化）
 *   confidence.shouldConclude / fallback.applyFallback
 *   machine.{createSession, view, computeNextSlot, buildReasons, runScoring, buildResult, apply}
 * 与后端差异：
 *   1) 去掉 model.complete 的 LLM 润色（结论理由由规则生成，结果完全一致）；
 *   2) 去掉病历文件上传解析（前端无法做服务端 PDF/Word 解析）；
 *   3) 会话状态存于内存（可选 localStorage 持久化见 index.html）。
 * 数据通过 fetch 从同源 ./data/*.json 加载，MiniSearch 由 vendor/minisearch.umd.js 提供。
 */
(function (global) {
  const MiniSearch = global.MiniSearch;

  const CONFIG = {
    triage: {
      maxFollowupRounds: 3,
      topN: 3,
      minSlotGain: 0.01,
      confidence: {
        concludeTop1: 0.45,
        concludeGap: 0.15,
        midTop1: 0.35,
        level1FallbackBelow: 0.12,
      },
      weights: {
        symptomMap: 0.3,
        bodyPart: 0.1,
        slots: 0.35,
        kbRetrieval: 0.15,
        historyPrior: 0.1,
        vectorScore: 0,
      },
    },
    kb: { retrieve: { topK: 8, minScore: 0.15 } },
  };

  // ---------- 中文分词（移植自 core/kb/tokenizer.js）----------
  const PUNCT =
    /[\s，。、；：？！“”‘’（）《》【】<>\[\](){}|/\\~,.;:!?@#$%^&*_+=`'"/\-—…·\u3000]+/g;

  function normalize(text) {
    if (!text) return '';
    return String(text)
      .toLowerCase()
      .replace(/\u3000/g, ' ')
      .replace(PUNCT, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function buildSynonymMap(synonyms) {
    const pairs = (synonyms && synonyms.pairs) || {};
    const map = {};
    for (const canon in pairs) {
      for (const v of pairs[canon]) map[v.toLowerCase()] = canon.toLowerCase();
    }
    const variants = Object.keys(map).sort((a, b) => b.length - a.length);
    return { map, variants };
  }

  function applySynonyms(text, synonym) {
    let out = text;
    for (const v of synonym.variants) {
      if (out.includes(v)) {
        const re = new RegExp(v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
        out = out.replace(re, synonym.map[v]);
      }
    }
    return out;
  }

  function createSegmenter(dict, synonyms) {
    const dictSorted = (dict || []).slice().sort((a, b) => b.length - a.length);
    const synonym = buildSynonymMap(synonyms);
    function segment(raw) {
      const norm = normalize(raw);
      if (!norm) return [];
      const text = applySynonyms(norm, synonym);
      const tokens = [];
      let i = 0;
      const n = text.length;
      while (i < n) {
        let matched = false;
        for (const term of dictSorted) {
          if (i + term.length <= n && text.substr(i, term.length) === term) {
            tokens.push(term);
            i += term.length;
            matched = true;
            break;
          }
        }
        if (!matched) {
          const ch = text[i];
          tokens.push(ch);
          if (i + 1 < n) tokens.push(ch + text[i + 1]);
          i += 1;
        }
      }
      return tokens.filter((t) => t && t !== ' ');
    }
    function tokenize(text) {
      return segment(text);
    }
    return { segment, tokenize, normalize, dictSet: new Set(dictSorted) };
  }

  // ---------- 检索（移植自 core/kb/indexer.js + retriever.js）----------
  function retrieverQuery(index, text, opts) {
    const topK = (opts && opts.topK) || 8;
    const minScore = (opts && opts.minScore) || 0.15;
    if (!text || !text.trim()) return [];
    let raw;
    try {
      raw = index.mini.search(text, { prefix: true, fuzzy: 0.1, combineWith: 'OR' });
    } catch (e) {
      raw = [];
    }
    if (!raw.length) return [];
    const maxScore = Math.max(...raw.map((r) => r.score));
    return raw
      .filter((r) => r.score >= minScore)
      .slice(0, topK)
      .map((r) => {
        const doc = index.docs[r.id];
        return {
          id: r.id,
          score: maxScore ? +(r.score / maxScore).toFixed(3) : +r.score.toFixed(3),
          rawScore: +r.score.toFixed(3),
          text: doc.text,
          source: doc.source,
          folder: doc.folder,
          chunkIndex: doc.chunkIndex,
          deptHits: index.deptHits[r.id] || [],
        };
      });
  }

  function retrieve(index, text, opts) {
    const results = retrieverQuery(index, text, opts);
    const deptScores = {};
    for (const r of results) {
      for (const code of r.deptHits || []) {
        if (!(code in deptScores) || r.score > deptScores[code]) {
          deptScores[code] = r.score;
        }
      }
    }
    return { results, deptScores };
  }

  // ---------- 运行时状态 ----------
  let KB = null;
  const sessions = new Map();

  function loadKBFromObjects(objs) {
    const { tree, depObj, idxObj, symMap, histObj, meta, kbIndex } = objs;
    const segmenter = createSegmenter(kbIndex.dict || [], kbIndex.synonyms || { pairs: {} });
    const mini = MiniSearch.loadJSON(JSON.stringify(kbIndex.mini), {
      fields: ['text'],
      storeFields: ['text', 'source', 'folder', 'chunkIndex'],
      tokenize: (s) => segmenter.tokenize(s),
      processTerm: (t) => t,
    });
    KB = {
      tree,
      departments: depObj.departments,
      replaced: depObj.replaced,
      indices: idxObj,
      symptomDeptMap: symMap.map,
      historyPrior: histObj.prior || {},
      builtAt: (meta || {}).builtAt,
      index: {
        mini,
        docs: mini.documents || [],
        deptHits: kbIndex.deptHits || [],
        query: (text, opts) => retrieverQuery(KB._idx, text, opts),
      },
    };
    KB._idx = { mini, deptHits: KB.index.deptHits, docs: KB.index.docs };
    return KB;
  }

  async function init(base) {
    const b = base || './data/';
    const get = (f) => fetch(b + f).then((r) => r.json());
    const [tree, depObj, idxObj, symMap, histObj, meta, kbIndex] = await Promise.all([
      get('question-tree.json'),
      get('departments.json'),
      get('dept-indices.json'),
      get('symptom-dept-map.json'),
      get('history-prior.json'),
      get('meta.json'),
      get('kb-index.json'),
    ]);
    return loadKBFromObjects({ tree, depObj, idxObj, symMap, histObj, meta, kbIndex });
  }

  // ---------- KB 访问器 ----------
  function get() {
    return KB;
  }
  function resolveDeptCode(name) {
    if (!name) return null;
    return KB.indices.nameToCode[String(name).toLowerCase()] || null;
  }
  function kbRetrieve(text) {
    return retrieve(KB._idx, text, {
      topK: CONFIG.kb.retrieve.topK,
      minScore: CONFIG.kb.retrieve.minScore,
    });
  }

  // ---------- question-tree 访问器 ----------
  function getBasics() {
    return KB.tree.basics || [];
  }
  function getBodyParts() {
    return KB.tree.bodyParts || [];
  }
  function getBodyPart(id) {
    return getBodyParts().find((b) => b.id === id) || null;
  }
  function getAllSymptoms() {
    return KB.tree.symptoms || [];
  }
  function getSymptom(id) {
    return getAllSymptoms().find((s) => s.id === id) || null;
  }
  function getSymptomsByBodyPart(bpId) {
    const bp = getBodyPart(bpId);
    if (!bp) return [];
    return (bp.symptoms || []).map(getSymptom).filter(Boolean);
  }
  function getSlots() {
    return KB.tree.slots || [];
  }
  function getSlot(id) {
    return getSlots().find((s) => s.id === id) || null;
  }
  function pendingSlotsForSymptom(symptomId, answeredSlotIds) {
    const sym = getSymptom(symptomId);
    if (!sym) return [];
    return (sym.slots || []).filter((sid) => !answeredSlotIds.has(sid)).map(getSlot).filter(Boolean);
  }

  // ---------- scorer（移植自 core/triage/scorer.js）----------
  function softmax(raw) {
    const keys = Object.keys(raw);
    if (!keys.length) return [];
    const max = Math.max(...keys.map((k) => raw[k]));
    const exp = {};
    let sum = 0;
    for (const k of keys) {
      exp[k] = Math.exp((raw[k] - max) / 0.7);
      sum += exp[k];
    }
    return keys
      .map((k) => ({ code: k, score: exp[k] / sum }))
      .sort((a, b) => b.score - a.score);
  }

  function score(st) {
    const kbState = get();
    const W = CONFIG.triage.weights;
    const deptMap = kbState.symptomDeptMap;
    const indices = kbState.indices;
    const historyPrior = kbState.historyPrior;
    const deptCodes = indices.level2;

    const s1 = {};
    for (const sid of st.symptomIds || []) {
      const m = deptMap[sid];
      if (!m) continue;
      for (const c in m) s1[c] = (s1[c] || 0) + m[c];
    }

    const s2 = {};
    for (const bp of st.bodyPartIds || []) {
      for (const c of deptCodes) {
        const d = kbState.departments.find((x) => x.code === c);
        if (d && (d.bodyParts || []).includes(bp)) s2[c] = 1;
      }
    }
    const s2sum = Object.values(s2).reduce((a, b) => a + b, 0);
    if (s2sum) for (const c in s2) s2[c] /= s2sum;

    const s3 = {};
    for (const slotId in st.answeredSlots || {}) {
      const ans = st.answeredSlots[slotId];
      const dw = ans && ans.deptWeights;
      if (!dw) continue;
      for (const name in dw) {
        const code = resolveDeptCode(name);
        if (!code) continue;
        s3[code] = (s3[code] || 0) + dw[name];
      }
    }

    let s4 = {};
    if (st.freeText && st.freeText.trim()) {
      const r = kbRetrieve(st.freeText);
      s4 = r.deptScores || {};
    }

    const s5 = {};
    for (const sid of st.symptomIds || []) {
      const hp = historyPrior[sid];
      if (!hp) continue;
      for (const c in hp) s5[c] = (s5[c] || 0) + hp[c];
    }

    const pedBoost = {};
    if (['infant', 'toddler', 'child', 'teen'].includes(st.ageBucket)) {
      for (const c of deptCodes) if (c === 'PED' || c.startsWith('PED_')) pedBoost[c] = 0.5;
    }

    const raw = {};
    for (const c of deptCodes) {
      let v =
        W.symptomMap * (s1[c] || 0) +
        W.bodyPart * (s2[c] || 0) +
        W.slots * (s3[c] || 0) +
        W.kbRetrieval * (s4[c] || 0) +
        W.historyPrior * (s5[c] || 0);
      v += pedBoost[c] || 0;
      if (v > 0) raw[c] = v;
    }

    if (st.gender === 'male') {
      for (const c of Object.keys(raw)) {
        const d = kbState.departments.find((x) => x.code === c);
        if (d && (d.parentCode === 'OBG' || d.code === 'OBG')) delete raw[c];
      }
    }

    const sumRaw = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
    const ranked = Object.keys(raw)
      .map((code) => ({ code, raw: raw[code], score: raw[code] / sumRaw }))
      .sort((a, b) => b.score - a.score)
      .map(({ code, score, raw: rawVal }) => {
        const d = kbState.departments.find((x) => x.code === code);
        return {
          code,
          name: d ? d.name : code,
          level: d ? d.level : 2,
          parentCode: d ? d.parentCode : null,
          score: +score.toFixed(3),
          raw: +rawVal.toFixed(3),
        };
      });

    return { ranked, signals: { s1, s2, s3, s4, s5, pedBoost } };
  }

  // ---------- confidence / fallback ----------
  function shouldConclude(ranked, ctx) {
    const c = CONFIG.triage.confidence;
    const top1 = ranked[0];
    const top2 = ranked[1];
    const reasons = [];
    if (!top1) return { conclude: true, reason: '无候选科室，转全科' };
    if (top1.score >= c.concludeTop1 && top2 && top1.score - top2.score >= c.concludeGap) {
      reasons.push('top1 显著高于次选');
    }
    if (ctx.followupCount >= CONFIG.triage.maxFollowupRounds) {
      reasons.push('已达最大追问轮次 ' + CONFIG.triage.maxFollowupRounds);
    }
    if (ctx.maxRemainingGain !== undefined && ctx.maxRemainingGain < CONFIG.triage.minSlotGain) {
      reasons.push('剩余槽位区分度不足');
    }
    if (ctx.forceConclude) reasons.push('患者主动结束');
    return { conclude: reasons.length > 0, reason: reasons.join('；') || '继续追问' };
  }

  function l1Code(d) {
    return d.parentCode || d.code;
  }

  function applyFallback(ranked) {
    const c = CONFIG.triage.confidence;
    const st = get();
    const topN = CONFIG.triage.topN;
    const list = ranked.slice(0, topN);
    const top1 = list[0];

    if (!top1) {
      const gp = st.departments.find((x) => x.code === 'GP');
      return {
        departments: [{ code: 'GP', name: gp ? gp.name : '全科医学科', level: 1, score: 0.4 }],
        fallbackNote: '无法定位，建议全科初诊',
        level: 1,
      };
    }

    if (top1.score < c.level1FallbackBelow) {
      const l1 = {};
      for (const d of ranked) {
        const p = l1Code(d);
        l1[p] = (l1[p] || 0) + d.score;
      }
      const merged = Object.entries(l1)
        .map(([code, score]) => {
          const dd = st.departments.find((x) => x.code === code);
          return { code, name: dd ? dd.name : code, level: 1, score };
        })
        .sort((a, b) => b.score - a.score);
      return {
        departments: merged.slice(0, topN),
        fallbackNote: '当前信息不足以精准定位二级科室，已按一级科室推荐',
        level: 1,
      };
    }

    if (list.length >= 2) {
      const l1set = new Set(list.slice(0, 3).map(l1Code));
      if (l1set.size === 1 && list[0].score - list[1].score < 0.05) {
        const p = [...l1set][0];
        const dd = st.departments.find((x) => x.code === p);
        return {
          departments: [{ code: p, name: dd ? dd.name : p, level: 1, score: list[0].score }],
          fallbackNote: `多个二级科室同属${dd ? dd.name : p}且难以区分，已直接推荐该一级科室`,
          level: 1,
        };
      }
    }

    const l1span = new Set(list.map(l1Code));
    if (l1span.size >= 3 && top1.score < 0.3) {
      const gp = st.departments.find((x) => x.code === 'GP');
      return {
        departments: [{ code: 'GP', name: gp ? gp.name : '全科医学科', level: 1, score: Math.max(top1.score, 0.4) }],
        fallbackNote: '症状涉及多个系统且不明确，建议先到全科门诊初诊评估',
        level: 1,
      };
    }

    return { departments: list, fallbackNote: '', level: 2 };
  }

  // ---------- machine ----------
  function createSession() {
    return {
      stage: 'INIT',
      basics: { age: null, gender: null },
      bodyPartIds: [],
      symptomIds: [],
      freeText: '',
      answeredSlots: {},
      askedSlots: [],
      followupCount: 0,
      history: [],
      result: null,
    };
  }

  function sessionState(session) {
    return {
      symptomIds: session.symptomIds,
      bodyPartIds: session.bodyPartIds,
      answeredSlots: session.answeredSlots,
      freeText: session.freeText,
      ageBucket: session.basics.age,
      gender: session.basics.gender,
    };
  }

  function computeNextSlot(session) {
    const ranked = score(sessionState(session)).ranked;
    const top3 = ranked.slice(0, 3).map((d) => d.code);
    if (!top3.length) return { slot: null, gain: 0 };

    const answeredIds = new Set(Object.keys(session.answeredSlots));
    const pending = new Set();
    for (const sid of session.symptomIds) {
      for (const s of pendingSlotsForSymptom(sid, answeredIds)) pending.add(s.id);
    }

    let best = null;
    let bestGain = 0;
    for (const slotId of pending) {
      const slot = getSlot(slotId);
      if (!slot) continue;
      if (slot.femaleOnly && session.basics.gender === 'male') continue;
      const contrib = top3.map(() => []);
      for (const opt of slot.options) {
        const dw = opt.deptWeights || {};
        const resolved = {};
        for (const n in dw) {
          const c = resolveDeptCode(n);
          if (c) resolved[c] = dw[n];
        }
        top3.forEach((c, i) => contrib[i].push(resolved[c] || 0));
      }
      let gain = 0;
      for (const arr of contrib) {
        const mean = arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
        gain += arr.reduce((a, b) => a + (b - mean) ** 2, 0) / (arr.length || 1);
      }
      if (gain > bestGain) {
        bestGain = gain;
        best = slot;
      }
    }
    return { slot: best, gain: bestGain };
  }

  function starsOf(scoreVal) {
    if (scoreVal >= 0.5) return 5;
    if (scoreVal >= 0.38) return 4;
    if (scoreVal >= 0.25) return 3;
    if (scoreVal >= 0.15) return 2;
    return 1;
  }

  function buildReasons(deptCode, session) {
    const reasons = [];
    const kbState = get();
    const symLabels = [];
    for (const sid of session.symptomIds) {
      const m = kbState.symptomDeptMap[sid];
      if (m && m[deptCode]) {
        const sym = getSymptom(sid);
        if (sym) symLabels.push(sym.label);
      }
    }
    if (symLabels.length) reasons.push(`您提到的「${symLabels.join('、')}」与该科室常见就诊原因相符`);

    const slotLabels = [];
    for (const slotId in session.answeredSlots) {
      const ans = session.answeredSlots[slotId];
      const dw = ans.deptWeights || {};
      for (const n in dw) {
        if (resolveDeptCode(n) === deptCode) {
          const slot = getSlot(slotId);
          slotLabels.push(`${slot ? slot.question.replace(/？$/, '') : '该项'}为「${ans.label}」`);
          break;
        }
      }
    }
    if (slotLabels.length) reasons.push(slotLabels.join('；'));

    const st = sessionState(session);
    if (st.freeText && st.freeText.trim()) {
      const r = kbRetrieve(st.freeText);
      if (r.deptScores[deptCode]) reasons.push('您补充的描述也与本病区相关');
    }
    return reasons;
  }

  function view(session) {
    switch (session.stage) {
      case 'INIT':
        return { stage: 'INIT', type: 'basics', title: '先了解一下基本情况', basics: getBasics() };
      case 'SELECT_BODY_PART':
        return {
          stage: 'SELECT_BODY_PART',
          type: 'bodyParts',
          title: '身体哪个部位不舒服？',
          bodyParts: getBodyParts(),
          selected: session.bodyPartIds,
        };
      case 'SELECT_SYMPTOMS': {
        const symptoms = [];
        for (const bp of session.bodyPartIds) {
          const part = getBodyPart(bp);
          if (!part) continue;
          symptoms.push({ bodyPartId: bp, bodyPartLabel: part.label, items: getSymptomsByBodyPart(bp) });
        }
        return {
          stage: 'SELECT_SYMPTOMS',
          type: 'symptoms',
          title: '具体是哪些表现？',
          groups: symptoms,
          selected: session.symptomIds,
        };
      }
      case 'FREE_TEXT':
        return { stage: 'FREE_TEXT', type: 'freeText', title: '还有想补充的情况吗？', optional: true };
      case 'FOLLOWUP': {
        const slotId = session.pendingSlot;
        const slot = getSlot(slotId);
        return {
          stage: 'FOLLOWUP',
          type: 'slot',
          title: slot ? slot.question : '再补充一点',
          slot: slot ? { id: slot.id, type: slot.type, options: slot.options } : null,
          round: session.followupCount + 1,
          maxRound: CONFIG.triage.maxFollowupRounds,
        };
      }
      case 'CONCLUDE':
        return { stage: 'CONCLUDE', type: 'result', result: session.result };
      default:
        return { stage: session.stage, type: 'unknown' };
    }
  }

  function buildResult(session, ranked, signals) {
    const fb = applyFallback(ranked);
    const depts = [];
    for (const d of fb.departments) {
      const reasons = buildReasons(d.code, session);
      const stars = starsOf(d.score);
      depts.push({
        code: d.code,
        name: d.name,
        level: d.level,
        stars,
        score: +d.score.toFixed(3),
        reasons,
        summary: reasons.join('；'),
      });
    }
    return {
      departments: depts,
      fallbackNote: fb.fallbackNote,
      level: fb.level,
      modelPolished: false,
      _raw: { ranked: ranked.slice(0, 5), signals, basics: session.basics, followupCount: session.followupCount },
    };
  }

  function runScoring(session) {
    const { ranked } = score(sessionState(session));
    const next = computeNextSlot(session);
    const conf = shouldConclude(ranked, {
      followupCount: session.followupCount,
      maxRemainingGain: next.gain,
    });
    if (conf.conclude || !next.slot) {
      session.result = buildResult(session, ranked, null);
      session.stage = 'CONCLUDE';
    } else {
      session.pendingSlot = next.slot.id;
      session.stage = 'FOLLOWUP';
    }
    return view(session);
  }

  function apply(session, msg) {
    const data = (msg && msg.data) || {};
    switch (session.stage) {
      case 'INIT': {
        session.basics.age = data.age || null;
        session.basics.gender = data.gender || null;
        session.stage = 'SELECT_BODY_PART';
        return view(session);
      }
      case 'SELECT_BODY_PART': {
        session.bodyPartIds = Array.isArray(data.bodyPartIds) ? data.bodyPartIds : [];
        session.stage = 'SELECT_SYMPTOMS';
        return view(session);
      }
      case 'SELECT_SYMPTOMS': {
        const ids = Array.isArray(data.symptomIds) ? data.symptomIds : [];
        session.symptomIds = Array.from(new Set(session.symptomIds.concat(ids)));
        if (data.done) session.stage = 'FREE_TEXT';
        return view(session);
      }
      case 'FREE_TEXT': {
        session.freeText = typeof data.freeText === 'string' ? data.freeText : '';
        return runScoring(session);
      }
      case 'FOLLOWUP': {
        const slotId = session.pendingSlot;
        const slot = getSlot(slotId);
        const value = data.value;
        const opt = slot && value !== undefined ? slot.options.find((o) => o.value === value) : null;
        if (slot && opt) {
          session.answeredSlots[slotId] = { value, label: opt.label, deptWeights: opt.deptWeights || {} };
          session.askedSlots.push(slotId);
        }
        if (data.finish) {
          session.result = buildResult(session, score(sessionState(session)).ranked, null);
          session.stage = 'CONCLUDE';
          return view(session);
        }
        session.followupCount += 1;
        return runScoring(session);
      }
      default:
        return view(session);
    }
  }

  // ---------- 对外 API ----------
  function start() {
    const sid = 's_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    const session = createSession();
    sessions.set(sid, session);
    return { sessionId: sid, view: view(session) };
  }

  function answer(sessionId, data) {
    const session = sessions.get(sessionId);
    if (!session) return { ok: false, error: '会话不存在或已过期' };
    const view2 = apply(session, { data });
    if (session.stage === 'CONCLUDE') {
      try {
        const raw = session.result && session.result._raw;
        if (raw) {
          const key = 'triage_record_' + sessionId;
          try {
            localStorage.setItem(
              key,
              JSON.stringify({
                at: new Date().toISOString(),
                basics: session.basics,
                symptoms: session.symptomIds,
                freeText: session.freeText,
                departments: (session.result.departments || []).map((d) => ({ name: d.name, score: d.score })),
              })
            );
          } catch (e) {}
        }
      } catch (e) {}
    }
    return { sessionId, view: view2 };
  }

  function result(sessionId) {
    const session = sessions.get(sessionId);
    if (!session) return { ok: false, error: '会话不存在或已过期' };
    return { sessionId, view: view(session) };
  }

  const api = { init, start, answer, result, CONFIG, _internal: { loadKBFromObjects, score, shouldConclude, applyFallback } };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (global) global.TriageEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
