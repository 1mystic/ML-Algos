(() => {
  const { ui, col, fmt, esc } = Lab;
  const SETS = {
    grocery: { items: ["bread", "milk", "eggs", "butter", "beer", "diapers", "chips", "cola"], rows: [[0, 1], [0, 5, 4, 2], [1, 5, 4, 7], [0, 1, 5, 4], [0, 1, 5, 7], [0, 1, 3], [4, 6, 7], [5, 4, 6], [0, 3, 2], [1, 5, 4, 6], [0, 1, 3, 2], [4, 6]] },
    breakfast: { items: ["coffee", "tea", "croissant", "toast", "jam", "juice", "yogurt"], rows: [[0, 2], [0, 2, 4], [1, 3, 4], [0, 2, 5], [1, 6], [0, 3, 4], [0, 2, 4], [1, 3, 4, 5], [0, 2, 6], [5, 6], [0, 2, 4, 5]] },
    random: null,
  };
  const key = (s) => s.join(",");
  const name = (s, items) => "{" + s.map((i) => items[i]).join(", ") + "}";
  function subsets(s, k) { const out = []; const rec = (st, cur) => { if (cur.length === k) { out.push(cur.slice()); return; } for (let i = st; i < s.length; i++) { cur.push(s[i]); rec(i + 1, cur); cur.pop(); } }; rec(0, []); return out; }

  function buildFrames(D, minSup, minConf) {
    const T = D.rows.map((r) => new Set(r)), N = T.length, items = D.items;
    const supp = (s) => T.filter((t) => s.every((i) => t.has(i))).length;
    const frames = [];
    const levels = []; // [{k, cands:[{s, sup, status}]}]
    const snap = (extra) => ({ levels: JSON.parse(JSON.stringify(levels)), ...extra });
    let Lk = null;
    for (let k = 1; k <= items.length; k++) {
      let cands;
      if (k === 1) cands = items.map((_, i) => [i]);
      else {
        const prev = Lk.map((c) => c.s);
        cands = [];
        for (let a = 0; a < prev.length; a++) for (let b = a + 1; b < prev.length; b++) {
          if (key(prev[a].slice(0, -1)) !== key(prev[b].slice(0, -1))) continue;
          cands.push(prev[a].concat([prev[b][k - 2]]).sort((x, y) => x - y));
        }
      }
      if (!cands.length) break;
      const lvl = { k, cands: cands.map((s) => ({ s, sup: null, status: "cand" })) };
      levels.push(lvl);
      frames.push({ phase: k === 1 ? "c1" : "join", major: true, arch: k === 1 ? ["db", "cand"] : ["freq", "join", "cand"], title: k === 1 ? `Level 1: every item is a candidate (${cands.length})` : `Level ${k}: join frequent ${k - 1}-itemsets → ${cands.length} candidates`, text: k === 1 ? "Apriori works level by level. It starts with single items, counts how many baskets contain each, and only extends the ones that are frequent." : `Two frequent ${k - 1}-itemsets that share their first ${k - 2} item${k - 2 === 1 ? "" : "s"} are merged into a ${k}-item candidate.`, tex: k === 1 ? "\\text{supp}(X)=\\frac{|\\{t\\in T: X\\subseteq t\\}|}{|T|}" : "C_k=\\{\\,a\\cup b:\\ a,b\\in L_{k-1},\\ a_{1..k-2}=b_{1..k-2}\\,\\}", st: snap({ level: k }) });
      if (k >= 3) {
        const prevKeys = new Set(Lk.map((c) => key(c.s)));
        let pruned = 0;
        for (const c of lvl.cands) if (subsets(c.s, k - 1).some((sub) => !prevKeys.has(key(sub)))) { c.status = "pruned"; pruned++; }
        frames.push({ phase: "prune", arch: ["cand", "prune"], title: `Apriori pruning: ${pruned} candidate${pruned === 1 ? "" : "s"} removed without scanning`, text: "If any (k−1)-subset of a candidate is infrequent, the candidate cannot be frequent either (support only shrinks as itemsets grow). These are crossed out before touching the data.", tex: "X\\subseteq Y\\;\\Rightarrow\\;\\text{supp}(Y)\\le\\text{supp}(X)", st: snap({ level: k }) });
      }
      const live = lvl.cands.filter((c) => c.status !== "pruned");
      live.forEach((c, idx) => {
        c.sup = supp(c.s);
        c.status = c.sup >= minSup ? "freq" : "rare";
        if (k <= 2 || live.length <= 6 || idx === live.length - 1) frames.push({ phase: "count", arch: ["db", "count"], title: `Count ${name(c.s, items)}: in ${c.sup} of ${N} baskets`, text: `Support ${c.sup}/${N} = <b>${(c.sup / N * 100).toFixed(0)}%</b> ${c.sup >= minSup ? "≥" : "<"} min support ${minSup}: ${c.sup >= minSup ? "<b>frequent</b>" : "dropped"}. Highlighted rows contain the itemset.`, st: snap({ level: k, focus: c.s }) });
      });
      Lk = lvl.cands.filter((c) => c.status === "freq");
      frames.push({ phase: "level", major: true, arch: ["count", "freq"], title: `L${k}: ${Lk.length} frequent ${k}-itemset${Lk.length === 1 ? "" : "s"}`, text: Lk.length ? Lk.map((c) => name(c.s, items)).join(", ") : "None: no larger itemset can be frequent, so the search stops.", st: snap({ level: k }), m: { cands: lvl.cands.length, freq: Lk.length } });
      if (Lk.length < 2) break;
    }
    // rules
    const rules = [];
    for (const lvl of levels) if (lvl.k >= 2) for (const c of lvl.cands.filter((x) => x.status === "freq")) {
      for (let a = 1; a < c.s.length; a++) for (const A of subsets(c.s, a)) {
        const B = c.s.filter((i) => !A.includes(i));
        const conf = c.sup / supp(A), lift = conf / (supp(B) / N);
        rules.push({ A, B, sup: c.sup / N, conf, lift, ok: conf >= minConf });
      }
    }
    rules.sort((p, q) => q.ok - p.ok || q.lift - p.lift);
    frames.push({ phase: "rules", major: true, arch: ["freq", "rules"], title: `Rules: ${rules.filter((r) => r.ok).length} of ${rules.length} meet confidence ≥ ${Math.round(minConf * 100)}%`, text: "Every frequent itemset is split into antecedent → consequent. Confidence is how often the rule holds when it applies; lift > 1 means the items co-occur more than independence would predict.", tex: "\\text{conf}(A\\Rightarrow B)=\\frac{\\text{supp}(A\\cup B)}{\\text{supp}(A)},\\qquad \\text{lift}=\\frac{\\text{conf}(A\\Rightarrow B)}{\\text{supp}(B)}", st: snap({ rules }) });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "apriori", fps: 1.6, plane: false,
      stageHTML: `<div class="html-stage" style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);gap:14px;align-items:start"><div><div class="sub-cap">transactions · click a cell to toggle an item</div><div data-r="tx" class="hscroll"></div></div><div><div class="sub-cap">itemset lattice, level by level</div><div data-r="lat"></div></div></div>`,
      legend: ui.legend([["var(--series-3)", "frequent"], ["var(--text-faint)", "infrequent"], ["var(--series-2)", "pruned (never counted)"]]),
      tools: ui.btn("replay", "replay", true),
      side: [{ title: "data", html: ui.select("ds", "baskets", [["grocery", "grocery (beer & diapers)"], ["breakfast", "breakfast café"], ["random", "random baskets"]], "grocery") }, { title: "thresholds", html: ui.range("sup", "min support (baskets)", 1, 8, 1, 3) + ui.range("conf", "min confidence", 0.1, 1, 0.05, 0.6) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "db", label: "transactions", sub: "baskets T", x: 58, y: 40, kind: "data" },
          { id: "cand", label: "candidates Cₖ", x: 186, y: 40, kind: "op" },
          { id: "prune", label: "prune", sub: "subset test", x: 310, y: 40, kind: "param" },
          { id: "count", label: "count support", sub: "scan T", x: 310, y: 110, kind: "op" },
          { id: "freq", label: "frequent Lₖ", sub: "supp ≥ min", x: 186, y: 110, kind: "loss" },
          { id: "join", label: "join", sub: "Lₖ ⋈ Lₖ → Cₖ₊₁", x: 58, y: 110, kind: "op" },
          { id: "rules", label: "rules A ⇒ B", sub: "conf, lift", x: 186, y: 168, kind: "out" },
        ],
        edges: [{ from: "db", to: "cand" }, { from: "cand", to: "prune" }, { from: "prune", to: "count" }, { from: "count", to: "freq" }, { from: "freq", to: "join" }, { from: "join", to: "cand", dash: true, label: "k + 1" }, { from: "freq", to: "rules" }],
      },
      phases: { c1: { label: "level 1", color: 3 }, join: { label: "join", color: 4 }, prune: { label: "prune", color: 1 }, count: { label: "count", color: 0 }, level: { label: "frequent set", color: 2 }, rules: { label: "rules", color: 2 } },
      panels: [{ id: "rules", title: "association rules", sub: "sorted by lift", wide: true }],
      charts: [{ id: "lv", title: "candidates vs frequent per level", custom: true }],
      onFrame: render,
    });
    let D = null;
    function loadSet() {
      const k = lab.val("ds");
      if (k === "random") {
        MLU.seedRng(Math.floor(Math.random() * 1e5));
        const items = ["apples", "bananas", "cheese", "dates", "eggs", "flour", "grapes"];
        D = { items, rows: Array.from({ length: 12 }, () => { const r = items.map((_, i) => i).filter((i) => MLU.rng() < (i < 3 ? 0.55 : 0.3)); return r.length ? r : [MLU.randInt(items.length)]; }) };
      } else D = JSON.parse(JSON.stringify(SETS[k]));
      replay();
    }
    let frames = [];
    function replay({ at = "start" } = {}) {
      frames = buildFrames(D, lab.val("sup"), lab.val("conf"));
      const lv = frames.filter((f) => f.phase === "level").map((f) => f.m);
      const el = lab.charts.lv.el; el.innerHTML = "";
      const b = Lab.bars(el, { w: 360, h: 130 });
      const items = [];
      lv.forEach((m, i) => { items.push({ label: `C${i + 1}`, value: m.cands, color: "var(--text-faint)", text: String(m.cands) }); items.push({ label: `L${i + 1}`, value: m.freq, color: "var(--series-3)", text: String(m.freq) }); });
      b.set(items);
      lab.load(frames, { at, play: at === "start" });
    }
    function drawTx(focus) {
      const el = lab.R("tx");
      const fs = focus ? new Set(focus) : null;
      el.innerHTML = `<table class="mini-table" style="min-width:300px"><tr><th>#</th>${D.items.map((it, j) => `<th style="text-align:center;${fs && fs.has(j) ? "color:var(--text)" : ""}" title="${esc(it)}">${esc(it.slice(0, 5))}</th>`).join("")}</tr>${D.rows.map((r, i) => {
        const has = new Set(r), hit = fs && focus.every((j) => has.has(j));
        return `<tr class="${hit ? "on" : ""}"><td>${i + 1}</td>${D.items.map((_, j) => `<td data-t="${i},${j}" style="text-align:center;cursor:pointer;color:${has.has(j) ? (fs && fs.has(j) ? "var(--series-4)" : "var(--text)") : "var(--border)"}">${has.has(j) ? "●" : "·"}</td>`).join("")}</tr>`;
      }).join("")}</table>`;
      el.querySelectorAll("td[data-t]").forEach((td) => td.onclick = () => {
        const [i, j] = td.dataset.t.split(",").map(Number);
        const r = D.rows[i];
        D.rows[i] = r.includes(j) ? r.filter((x) => x !== j) : r.concat([j]).sort((a, b) => a - b);
        replay({ at: "end" });
      });
    }
    function drawLattice(st) {
      const el = lab.R("lat"); el.innerHTML = "";
      const lv = st.levels;
      const rowH = 54, W = 460;
      const H = Math.max(80, lv.length * rowH + 20);
      const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("class", "arch-svg");
      const pos = new Map();
      lv.forEach((L, r) => {
        const n = L.cands.length, w = Math.min(96, (W - 20) / Math.max(1, n) - 6);
        L.cands.forEach((c, i) => {
          const x = 10 + ((W - 20) / n) * (i + 0.5), y = 18 + r * rowH;
          pos.set(key(c.s), [x, y]);
          if (r > 0) for (const sub of subsets(c.s, c.s.length - 1)) { const p = pos.get(key(sub)); if (p) svg.insert("line", ":first-child").attr("x1", p[0]).attr("y1", p[1] + 11).attr("x2", x).attr("y2", y - 11).attr("stroke", "var(--border)").attr("stroke-width", 0.8); }
          const g = svg.append("g").attr("transform", `translate(${x},${y})`);
          const on = st.focus && key(st.focus) === key(c.s);
          const fill = c.status === "freq" ? "color-mix(in srgb, var(--series-3) 28%, var(--bg-inset))" : c.status === "pruned" ? "color-mix(in srgb, var(--series-2) 14%, var(--bg-inset))" : "var(--bg-inset)";
          g.append("rect").attr("x", -w / 2).attr("y", -11).attr("width", w).attr("height", 22).attr("rx", 5).attr("fill", fill).attr("stroke", on ? "var(--series-4)" : c.status === "freq" ? "var(--series-3)" : "var(--border)").attr("stroke-width", on ? 2.4 : 1).attr("opacity", c.status === "rare" ? 0.5 : 1);
          const label = c.s.map((i) => D.items[i].slice(0, n > 8 ? 2 : 4)).join("+");
          g.append("text").attr("text-anchor", "middle").attr("y", c.sup != null ? -1 : 4).style("font-size", "8.5px").attr("fill", "var(--text)").text(label);
          if (c.sup != null) g.append("text").attr("text-anchor", "middle").attr("y", 8).style("font-size", "7.5px").attr("fill", "var(--text-faint)").text(`supp ${c.sup}`);
          if (c.status === "pruned") g.append("line").attr("x1", -w / 2 + 4).attr("x2", w / 2 - 4).attr("y1", 0).attr("y2", 0).attr("stroke", "var(--series-2)").attr("stroke-width", 1.4);
        });
      });
    }
    function render(f) {
      const st = f.st;
      drawTx(st.focus);
      drawLattice(st);
      const R = st.rules;
      lab.R("panel-rules").innerHTML = R ? `<table class="mini-table"><tr><th>rule</th><th>support</th><th>confidence</th><th>lift</th></tr>${R.slice(0, 14).map((r) => `<tr class="${r.ok ? "ok" : "bad"}"><td>${name(r.A, D.items)} ⇒ ${name(r.B, D.items)}</td><td>${(r.sup * 100).toFixed(0)}%</td><td>${(r.conf * 100).toFixed(0)}%</td><td>${r.lift.toFixed(2)}</td></tr>`).join("")}</table>` : `<div class="lab-card-note">Rules are generated once all frequent itemsets are known.</div>`;
      const nf = st.levels.reduce((s, L) => s + L.cands.filter((c) => c.status === "freq").length, 0);
      const counted = st.levels.reduce((s, L) => s + L.cands.filter((c) => c.sup != null).length, 0);
      const brute = Math.pow(2, D.items.length) - 1;
      lab.readout(`<div class="kv"><span>baskets × items</span><b>${D.rows.length} × ${D.items.length}</b><span>itemsets counted</span><b>${counted}</b><span>vs brute force</span><b>${brute}</b><span>frequent so far</span><b>${nf}</b></div>`);
    }
    lab.watch(["ds"], () => loadSet());
    lab.watch(["sup", "conf"], () => replay({ at: "end" }), { format: { conf: (v) => Math.round(v * 100) + "%" } });
    lab.q('[data-b="replay"]').onclick = () => replay();
    loadSet();
    return lab.destroy;
  }

  MLApp.register({
    id: "apriori",
    name: "Apriori (Association Rules)",
    category: "Unsupervised - Anomaly & Association",
    tagline: "frequent itemsets → if-then rules",
    isNew: true,
    description: "Market-basket analysis: find item combinations that appear together often, then turn them into rules like {diapers} ⇒ {beer}. Watch candidates get generated level by level, pruned by the Apriori property, counted against the baskets, and scored by confidence and lift.",
    info: {
      type: "Unsupervised - Association rule mining over transactional data. Level-wise, breadth-first search.",
      scenario: "Transaction or event logs where you want co-occurrence patterns: retail baskets, web sessions, symptoms, software features used together.",
      inputs: "A set of transactions, each a set of items; minimum support and minimum confidence thresholds.",
      output: "Frequent itemsets with their support, and association rules A ⇒ B with support, confidence and lift.",
      intuition: {
        definition: "Count which items appear together in many baskets. The trick that makes it feasible: an itemset can only be frequent if all of its subsets are, so you build up from frequent single items to pairs to triples and never count combinations that contain an infrequent part.",
        steps: ["Count every single item; keep those above min support (L₁).", "Join frequent k-itemsets to form (k+1)-item candidates.", "Discard candidates with any infrequent subset (Apriori pruning).", "Scan the data to count the survivors; keep the frequent ones.", "Repeat until no candidates remain; derive rules from the frequent itemsets."],
        applications: ["Retail market-basket analysis and store layout", "Cross-selling and 'frequently bought together' widgets", "Web usage mining and clickstream patterns", "Medical co-morbidity and adverse-event patterns"],
      },
      math: [
        { title: "Support", formula: "supp(X) = |{t ∈ T : X ⊆ t}| / |T|", note: "The fraction of transactions containing every item of X." },
        { title: "Anti-monotonicity (Apriori property)", formula: "X ⊆ Y ⇒ supp(Y) ≤ supp(X)", note: "Adding items can only lose transactions. So every subset of a frequent itemset is frequent, and any superset of an infrequent one is infrequent." },
        { title: "Candidate generation", formula: "C_{k+1} = {a ∪ b : a, b ∈ L_k, share first k−1 items}", note: "Followed by pruning candidates that have an infrequent k-subset." },
        { title: "Confidence", formula: "conf(A ⇒ B) = supp(A ∪ B) / supp(A)", note: "An estimate of P(B | A)." },
        { title: "Lift", formula: "lift(A ⇒ B) = conf(A ⇒ B) / supp(B)", note: "> 1 means positive association beyond chance; = 1 independence; < 1 negative association." },
      ],
      pipeline: [{ label: "Count items" }, { label: "Join → Cₖ" }, { label: "Prune", note: "subset test" }, { label: "Count support" }, { label: "Rules", note: "conf, lift", accent: "green" }],
      decisionFunction: { text: "keep X if supp(X) ≥ min_sup;  keep A ⇒ B if conf ≥ min_conf (and often lift > 1)", mechanism: "Pure counting and thresholding: there is no model to train, only a search over the itemset lattice made tractable by pruning." },
      lossFunction: { text: "No loss: an exact, exhaustive search above the support threshold", mechanism: "Apriori returns every itemset meeting the support threshold; the only approximations are the thresholds you choose." },
      optimization: [{ title: "Worst case", formula: "O(2^|I|) itemsets", note: "Pruning makes typical runs far cheaper, but very low support thresholds can still explode." }, { title: "Multiple scans", formula: "one pass over T per level", note: "The main cost on large data; FP-Growth avoids candidate generation with a compressed prefix tree." }, { title: "Hash trees / tid-lists", formula: "faster support counting", note: "Eclat uses vertical tid-list intersections instead of scans." }],
      assumptions: [{ name: "Transactions are sets", why: "Quantities and order are ignored.", check: "Use sequential pattern mining for order." }, { name: "Support threshold is meaningful", why: "Rare but important patterns fall below it.", check: "Lower it carefully, or use lift/leverage-based mining." }],
      hyperparameters: [{ name: "min_support", range: "0.1% - 20%", increasing: "Fewer, more common itemsets; much faster.", strategy: "Start high and lower until the output is useful but manageable." }, { name: "min_confidence", range: "30% - 90%", increasing: "Fewer, more reliable rules.", strategy: "Combine with a lift > 1 filter." }, { name: "max_len", range: "2 - 5", increasing: "Longer itemsets allowed.", strategy: "Cap to keep rules readable." }],
      metrics: ["Support", "Confidence", "Lift", "Leverage and conviction"],
      typicalUses: ["Basket analysis", "Recommendation seeds", "Pattern discovery in logs"],
      diagnostics: ["Sort rules by lift, not confidence: high confidence can simply mean B is very common.", "Check rule redundancy: many rules are subsets of others.", "Validate on a later time period."],
      advantages: ["Exact and easy to explain.", "Output (rules) is directly human-readable.", "Pruning principle is simple and effective."],
      limitations: [{ name: "Many passes over the data", note: "one per level", fix: "FP-Growth or Eclat." }, { name: "Candidate explosion at low support", note: "", fix: "raise support, cap max_len." }, { name: "Spurious rules", note: "confidence ignores the base rate of B", fix: "filter by lift, test significance." }],
      alternatives: [{ name: "FP-Growth", when: "Large datasets; no candidate generation, two passes." }, { name: "Eclat", when: "Vertical data layout, dense datasets." }, { name: "Collaborative filtering", when: "You want personalised recommendations rather than global rules." }],
      pitfalls: [{ problem: "Thousands of trivial rules", solution: "Raise thresholds and filter by lift." }, { problem: "Rule with 90% confidence but lift 1.0", solution: "B is in 90% of baskets anyway; the rule says nothing." }, { problem: "Runs forever", solution: "Support too low; the lattice exploded." }],
      quickRef: [{ name: "Support", formula: "|{t ⊇ X}| / |T|" }, { name: "Confidence", formula: "supp(A∪B) / supp(A)" }, { name: "Lift", formula: "conf / supp(B)" }, { name: "Apriori property", formula: "X ⊆ Y ⇒ supp(Y) ≤ supp(X)" }],
      code: `import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

te = TransactionEncoder()
df = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
freq = apriori(df, min_support=0.25, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.6)
rules.sort_values("lift", ascending=False)`,
      whyChain: [
        { q: "Why can Apriori prune without counting?", a: "Support is anti-monotone: every transaction containing a superset also contains each of its subsets. So if any subset is below the threshold, the candidate must be too." },
        { q: "Why is lift better than confidence?", a: "Confidence P(B|A) is high whenever B is common, even if A tells you nothing. Lift divides by P(B), so it measures how much A raises the chance of B." },
        { q: "Why did FP-Growth replace Apriori in practice?", a: "Apriori scans the database once per level and can generate huge candidate sets. FP-Growth compresses the data into a prefix tree in two scans and mines it recursively with no candidates." },
      ],
      workedExample: {
        setup: "Baskets: {milk, bread}, {milk, diapers, beer}, {bread, diapers, beer}, {milk, bread, diapers, beer}. min support 2.",
        steps: ["L₁: milk 3, bread 3, diapers 3, beer 3 (all frequent).", "C₂ counts: {diapers, beer} = 3, {milk, bread} = 2, others ≥ 2 kept as counted.", "Rule diapers ⇒ beer: conf = 3/3 = 100%.", "lift = 1.0 / (3/4) = 1.33 > 1: positive association."],
        result: "{diapers} ⇒ {beer} with confidence 100% and lift 1.33",
      },
    },
    mount,
  });
})();
