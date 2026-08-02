// App shell: registry, sidebar navigation, hash router.
// Each algorithm file calls MLApp.register({...}) when it loads.
const MLApp = (() => {
  const registry = [];
  const order = [
    "Supervised - Regression",
    "Supervised - Classification",
    "Supervised - Trees & Ensembles",
    "Unsupervised - Clustering",
    "Unsupervised - Dimensionality Reduction",
    "Deep Learning",
    "Reinforcement Learning",
    "Supplementary",
  ];
  const CATEGORY_META = {
    "Supervised - Regression": {
      color: "var(--pastel-green)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17 L9 11 L13 14 L20 5"/><circle cx="4" cy="17" r="1.3" fill="currentColor" stroke="none"/><circle cx="9" cy="11" r="1.3" fill="currentColor" stroke="none"/><circle cx="13" cy="14" r="1.3" fill="currentColor" stroke="none"/><circle cx="20" cy="5" r="1.3" fill="currentColor" stroke="none"/></svg>`,
    },
    "Supervised - Classification": {
      color: "var(--pastel-red)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="21" x2="21" y2="3"/><circle cx="7" cy="8" r="1.5" fill="currentColor" stroke="none"/><circle cx="11" cy="6" r="1.5" fill="currentColor" stroke="none"/><circle cx="9" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="15" cy="18" r="1.5"/><circle cx="18" cy="14" r="1.5"/><circle cx="14" cy="20" r="1.5"/></svg>`,
    },
    "Supervised - Trees & Ensembles": {
      color: "var(--series-4)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="4" r="1.6" fill="currentColor" stroke="none"/><line x1="12" y1="5.6" x2="12" y2="9"/><line x1="12" y1="9" x2="6" y2="14"/><line x1="12" y1="9" x2="18" y2="14"/><circle cx="6" cy="15.6" r="1.5"/><circle cx="18" cy="15.6" r="1.5"/><line x1="6" y1="17" x2="4" y2="21"/><line x1="6" y1="17" x2="8" y2="21"/><line x1="18" y1="17" x2="16" y2="21"/><line x1="18" y1="17" x2="20" y2="21"/></svg>`,
    },
    "Unsupervised - Clustering": {
      color: "var(--series-1)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="9" r="5"/><circle cx="16" cy="14" r="4"/><circle cx="7" cy="17" r="3"/></svg>`,
    },
    "Unsupervised - Dimensionality Reduction": {
      color: "var(--series-5)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="20" x2="21" y2="20"/><circle cx="6" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="12" cy="4" r="1.3" fill="currentColor" stroke="none"/><circle cx="17" cy="9" r="1.3" fill="currentColor" stroke="none"/><line x1="6" y1="6" x2="6" y2="20" stroke-dasharray="2 2"/><line x1="12" y1="4" x2="12" y2="20" stroke-dasharray="2 2"/><line x1="17" y1="9" x2="17" y2="20" stroke-dasharray="2 2"/></svg>`,
    },
    "Deep Learning": {
      color: "var(--series-6)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="4" cy="7" r="1.5"/><circle cx="4" cy="17" r="1.5"/><circle cx="12" cy="4" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="20" r="1.5"/><circle cx="20" cy="9" r="1.5"/><circle cx="20" cy="16" r="1.5"/><line x1="4" y1="7" x2="12" y2="4"/><line x1="4" y1="7" x2="12" y2="12"/><line x1="4" y1="17" x2="12" y2="12"/><line x1="4" y1="17" x2="12" y2="20"/><line x1="12" y1="4" x2="20" y2="9"/><line x1="12" y1="12" x2="20" y2="9"/><line x1="12" y1="12" x2="20" y2="16"/><line x1="12" y1="20" x2="20" y2="16"/></svg>`,
    },
    "Supplementary": {
      color: "var(--series-3)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H12v18H6.5A2.5 2.5 0 0 0 4 18.5Z"/><path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H12v18h5.5a2.5 2.5 0 0 1 2.5 2.5Z" transform="translate(0,-1.5)"/><path d="M12 3v18"/></svg>`,
    },
    "Reinforcement Learning": {
      color: "var(--series-7)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.2"/><rect x="14" y="3" width="7" height="7" rx="1.2"/><rect x="3" y="14" width="7" height="7" rx="1.2"/><rect x="14" y="14" width="7" height="7" rx="1.2" fill="currentColor" stroke="none" opacity="0.45"/><path d="M10 6.5 H14 M17.5 10 V14 M6.5 14 V17.5"/></svg>`,
    },
  };

  let active = null;
  let activeCleanup = null;
  // Section slug to reveal after the next algorithm render, set when a search
  // result points at a specific passage rather than the page as a whole.
  let pendingFocus = null;

  function register(algo) { registry.push(algo); }

  function byId(id) { return registry.find((a) => a.id === id); }

  // Sidebar groups start collapsed so the rail stays readable. Navigating to an
  // algorithm auto-opens its group, but that is only a default: the user can
  // collapse it again like any other, and every choice is remembered.
  const openGroups = (() => {
    try { return new Set(JSON.parse(localStorage.getItem("ml-open-groups") || "[]")); }
    catch (e) { return new Set(); }
  })();
  // Categories the user explicitly collapsed, so auto-open does not undo them.
  const closedGroups = new Set();

  function persistOpenGroups() {
    try { localStorage.setItem("ml-open-groups", JSON.stringify([...openGroups])); } catch (e) {}
  }

  function renderSidebar() {
    const nav = document.getElementById("sidebar-nav");
    MLU.clearNode(nav);
    const activeAlgo = active ? byId(active) : null;
    const activeCat = activeAlgo ? activeAlgo.category : null;

    for (const cat of order) {
      const items = registry.filter((a) => a.category === cat);
      if (!items.length) continue;
      const meta = CATEGORY_META[cat] || { color: "var(--text)", icon: "" };
      const isOpen = openGroups.has(cat) || (cat === activeCat && !closedGroups.has(cat));

      const group = document.createElement("div");
      group.className = "sidebar-group" + (isOpen ? " open" : "");
      group.style.setProperty("--cat-color", meta.color);

      const title = document.createElement("button");
      title.className = "sidebar-group-title";
      title.type = "button";
      title.setAttribute("aria-expanded", String(isOpen));
      title.innerHTML = `
        <span class="sg-icon">${meta.icon}</span>
        <span class="sg-label">${cat}</span>
        <span class="sg-count">${items.length}</span>
        <span class="sg-chevron" aria-hidden="true">›</span>`;
      title.onclick = () => {
        if (isOpen) { openGroups.delete(cat); closedGroups.add(cat); }
        else { openGroups.add(cat); closedGroups.delete(cat); }
        persistOpenGroups();
        renderSidebar();
      };
      group.appendChild(title);

      const body = document.createElement("div");
      body.className = "sidebar-group-body";
      for (const algo of items) {
        const item = document.createElement("div");
        item.className = "nav-item" + (algo.id === active ? " active" : "");
        item.tabIndex = 0;
        item.innerHTML = `<span class="name">${algo.name}</span><span class="tag">${algo.tagline}</span>`;
        const go = () => { location.hash = "#/" + algo.id; };
        item.onclick = go;
        item.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } };
        body.appendChild(item);
      }
      group.appendChild(body);
      nav.appendChild(group);
    }
  }

  function renderHome() {
    active = null;
    const main = document.getElementById("main");
    main.innerHTML = `
      <div class="main-header">
        <h1>ML Algorithms - Interactive Lab</h1>
        <p>An interactive reference for classical machine learning. Every algorithm below is implemented from scratch in JavaScript, so you can click, drag and step through it live, and each one carries a full study guide covering the mathematics, assumptions, tuning, pitfalls and interview questions. Pick one from the sidebar, search above, or start with a card below.</p>
      </div>
      <div class="home-categories">
        ${order.map((cat) => {
          const items = registry.filter((a) => a.category === cat);
          if (!items.length) return "";
          const meta = CATEGORY_META[cat] || { color: "var(--text)", icon: "" };
          return `
          <div class="category-card" style="--cat-color:${meta.color}">
            <div class="category-header">
              <div class="category-icon">${meta.icon}</div>
              <div>
                <div class="category-title">${cat}</div>
                <div class="category-count">${items.length} algorithm${items.length > 1 ? "s" : ""}</div>
              </div>
            </div>
            <div class="algo-stack">
              ${items.map((a) => `
                <div class="algo-row" data-id="${a.id}" tabindex="0" role="button">
                  <div class="algo-icon">${meta.icon}</div>
                  <div class="algo-info">
                    <div class="algo-name">${a.name}</div>
                    <div class="algo-desc">${a.description}</div>
                  </div>
                </div>`).join("")}
            </div>
          </div>`;
        }).join("")}
      </div>
    `;
    main.querySelectorAll(".algo-row").forEach((el) => {
      const go = () => { location.hash = "#/" + el.dataset.id; };
      el.onclick = go;
      el.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } };
    });
    renderSidebar();
  }

  const PRINT_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9V3h12v6"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="7" rx="1"/></svg>`;

  // Supplementary pages have no interactive demo, so they skip the tab bar and
  // render their prose straight into the reference container.
  function renderGuide(algo) {
    const main = document.getElementById("main");
    main.innerHTML = `
      <div class="main-header">
        <h1>${algo.name}</h1>
        <p>${algo.description}</p>
      </div>
      <div class="guide-bar">
        <span class="guide-kicker">Supplementary guide</span>
        <button class="print-btn" id="print-btn" title="Print this guide (A4)">${PRINT_ICON} print</button>
      </div>
      <div id="tab-ref"></div>
    `;
    renderSidebar();
    renderGuideBody(document.getElementById("tab-ref"), algo.guide);
    const printBtn = document.getElementById("print-btn");
    if (printBtn) printBtn.onclick = () => window.print();
    applyPendingFocus();
  }

  function applyPendingFocus() {
    if (!pendingFocus) return;
    const target = pendingFocus;
    pendingFocus = null;
    const el = document.querySelector(`[data-section="${target}"]`);
    if (el) {
      el.scrollIntoView({ block: "start", behavior: "smooth" });
      el.classList.add("section-flash");
      setTimeout(() => el.classList.remove("section-flash"), 1600);
    }
  }

  function renderAlgo(id) {
    const algo = byId(id);
    if (!algo) { renderHome(); return; }
    if (typeof activeCleanup === "function") { try { activeCleanup(); } catch (e) {} }
    activeCleanup = null;
    active = id;
    if (algo.kind === "guide") { renderGuide(algo); return; }
    const main = document.getElementById("main");
    main.innerHTML = `
      <div class="main-header">
        <h1>${algo.name}</h1>
        <p>${algo.description}</p>
      </div>
      <div class="tabs">
        <div class="tab active" data-tab="play">Playground</div>
        <div class="tab" data-tab="ref">Reference</div>
        <button class="print-btn" id="print-btn" title="Print the reference sheet (A4)">${PRINT_ICON} print</button>
      </div>
      <div id="tab-play"></div>
      <div id="tab-ref" style="display:none"></div>
    `;
    renderSidebar();
    const playRoot = document.getElementById("tab-play");
    const cleanup = algo.mount(playRoot, MLU);
    if (typeof cleanup === "function") activeCleanup = cleanup;
    if (algo.info) renderInfo(document.getElementById("tab-ref"), algo.info);

    function showTab(which) {
      main.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === which));
      document.getElementById("tab-play").style.display = which === "play" ? "" : "none";
      document.getElementById("tab-ref").style.display = which === "ref" ? "" : "none";
    }

    main.querySelectorAll(".tab").forEach((tab) => {
      tab.onclick = () => showTab(tab.dataset.tab);
    });

    // The print stylesheet always lays out the reference sheet, so switch to it
    // first: printing what you cannot see is disorienting.
    const printBtn = document.getElementById("print-btn");
    if (printBtn) printBtn.onclick = () => { showTab("ref"); window.print(); };

    // Arriving from a content search result: open the reference tab and jump to
    // the matching section.
    if (pendingFocus) { showTab("ref"); applyPendingFocus(); }
  }

  // ---- reference-tab rendering helpers -------------------------------------
  const list = (arr) => `<ul class="info-list">${arr.map((x) => `<li>${x}</li>`).join("")}</ul>`;

  // A titled full-width section of the reference page. Returns "" when the
  // section has no content, so every part of the schema below is optional and
  // algorithms can be migrated to the expanded format one at a time.
  // `slug` matches the SECTIONS keys below so a search hit can scroll here.
  function section(num, title, body, slug) {
    if (!body) return "";
    return `<section class="ref-section"${slug ? ` data-section="${slug}"` : ""}>
      <h2 class="ref-h2"><span class="ref-num">${num}</span>${title}</h2>
      ${body}
    </section>`;
  }

  // Generic table: headers + rows of cells. First column is emphasised.
  function table(headers, rows) {
    if (!rows || !rows.length) return "";
    return `<div class="hscroll"><table class="ref-table">
      <thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>`;
  }

  // Left-to-right flow of labelled boxes, used in place of a mermaid diagram
  // so the page keeps working with no extra dependency.
  function flow(nodes) {
    if (!nodes || !nodes.length) return "";
    return `<div class="flow">${nodes.map((n, i) => `
      ${i ? `<div class="flow-arrow" aria-hidden="true">→</div>` : ""}
      <div class="flow-node${n.accent ? " accent-" + n.accent : ""}">
        <div class="flow-label">${n.label}</div>
        ${n.note ? `<div class="flow-note">${n.note}</div>` : ""}
      </div>`).join("")}</div>`;
  }

  // Numbered derivation: each step is a formula with a short explanation.
  function steps(items) {
    if (!items || !items.length) return "";
    return `<div class="math-steps">${items.map((s, i) => `
      <div class="math-step">
        <div class="math-step-head"><span class="math-step-num">${i + 1}</span>${s.title}</div>
        ${s.formula ? `<p class="formula">${s.formula}</p>` : ""}
        ${s.note ? `<p class="math-step-note">${s.note}</p>` : ""}
      </div>`).join("")}</div>`;
  }

  // Examiner-style Q&A chain - the "why does that follow?" drill-down.
  function whyChain(items) {
    if (!items || !items.length) return "";
    return `<div class="why-chain">${items.map((qa) => `
      <div class="why-item">
        <div class="why-q">${qa.q}</div>
        <div class="why-a">${qa.a}</div>
      </div>`).join("")}</div>`;
  }

  // A decision chart: one question, several labelled outcomes. Outcomes wrap
  // onto their own rows on narrow screens rather than scrolling sideways.
  function branch(node) {
    if (!node) return "";
    return `<div class="branch">
      <div class="branch-q">${node.q}</div>
      <div class="branch-arms">${(node.arms || []).map((a) => `
        <div class="branch-arm${a.accent ? " accent-" + a.accent : ""}">
          <div class="branch-when">${a.when}</div>
          <div class="branch-then">${a.then}</div>
          ${a.note ? `<div class="branch-note">${a.note}</div>` : ""}
        </div>`).join("")}</div>
    </div>`;
  }

  // A phased pipeline: named stages, each holding an ordered list of steps.
  // Stages sit side by side on desktop and stack on mobile.
  function lanes(stages) {
    if (!stages || !stages.length) return "";
    return `<div class="lanes">${stages.map((s, i) => `
      <div class="lane${s.accent ? " accent-" + s.accent : ""}">
        <div class="lane-head"><span class="lane-num">${i + 1}</span>${s.name}</div>
        ${s.note ? `<div class="lane-note">${s.note}</div>` : ""}
        <ul class="lane-items">${(s.items || []).map((it) => `<li>${it}</li>`).join("")}</ul>
      </div>`).join("")}</div>`;
  }

  // A short highlighted aside, tinted green for guidance and red for warnings.
  function callout(c) {
    if (!c) return "";
    return `<div class="callout accent-${c.tone || "green"}">
      <div class="callout-title">${c.title}</div>
      <div class="callout-text">${c.text}</div>
    </div>`;
  }

  function codeBlock(src) {
    if (!src) return "";
    const esc = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return `<pre class="code-block"><code>${esc}</code></pre>`;
  }

  // Guide pages describe their content as an ordered list of typed blocks, so a
  // page can mix prose, tables and diagrams without a fixed schema.
  function renderBlock(b) {
    if (typeof b === "string") return `<p class="guide-p">${b}</p>`;
    if (b.p) return `<p class="guide-p">${b.p}</p>`;
    if (b.list) return list(b.list);
    if (b.olist) return `<ol class="info-list">${b.olist.map((x) => `<li>${x}</li>`).join("")}</ol>`;
    if (b.table) return table(b.table.headers, b.table.rows);
    if (b.steps) return steps(b.steps);
    if (b.flow) return `<div class="flow-wrap">${b.title ? `<div class="flow-title">${b.title}</div>` : ""}${flow(b.flow)}</div>`;
    if (b.lanes) return lanes(b.lanes);
    if (b.branch) return branch(b.branch);
    if (b.callout) return callout(b.callout);
    if (b.qa) return whyChain(b.qa);
    if (b.code) return codeBlock(b.code);
    if (b.cards) {
      return `<div class="info-grid">${b.cards.map((c) => `
        <div class="info-block${c.span ? " span2" : ""}">
          <h4>${c.name}</h4>
          ${c.list ? list(c.list) : `<p>${c.note}</p>`}
        </div>`).join("")}</div>`;
    }
    return "";
  }

  function renderGuideBody(container, guide) {
    let n = 0;
    container.innerHTML = (guide || []).map((sec) =>
      section(++n, sec.title, (sec.body || []).map(renderBlock).join(""), sec.slug)
    ).join("");
  }

  function renderInfo(container, info) {
    const i = info;
    const num = (() => { let n = 0; return () => ++n; })();

    // 1 - Core concept: definition, intuition steps, real-world uses.
    const overview = `
      <div class="info-grid">
        ${i.intuition && i.intuition.definition ? `<div class="info-block span2 lede"><p>${i.intuition.definition}</p></div>` : ""}
        <div class="info-block"><h4>Type of algorithm</h4><p>${i.type}</p></div>
        <div class="info-block"><h4>When it's used</h4><p>${i.scenario}</p></div>
        <div class="info-block"><h4>Inputs</h4><p>${i.inputs}</p></div>
        <div class="info-block"><h4>Output</h4><p>${i.output}</p></div>
        ${i.intuition && i.intuition.steps ? `<div class="info-block"><h4>Core intuition</h4>${list(i.intuition.steps)}</div>` : ""}
        ${i.intuition && i.intuition.applications ? `<div class="info-block"><h4>Real-world applications</h4>${list(i.intuition.applications)}</div>` : ""}
      </div>`;

    const mathBody = [
      steps(i.math),
      i.pipeline ? `<div class="flow-wrap"><div class="flow-title">Pipeline</div>${flow(i.pipeline)}</div>` : "",
      `<div class="info-grid"><div class="info-block span2">
        <h4>Decision function &amp; mechanism</h4>
        <p class="formula">${i.decisionFunction.text}</p>
        <p>${i.decisionFunction.mechanism}</p>
        ${i.decisionFunction.plot ? `<div class="plot-slot" id="plot-decision"></div><div class="plot-caption">${i.decisionFunction.plot.caption || ""}</div>` : ""}
      </div></div>`,
    ].join("");

    const lossBody = [
      `<div class="info-grid"><div class="info-block span2">
        <p class="formula">${i.lossFunction.text}</p>
        <p>${i.lossFunction.mechanism}</p>
        ${i.lossFunction.plot ? `<div class="plot-slot" id="plot-loss"></div><div class="plot-caption">${i.lossFunction.plot.caption || ""}</div>` : ""}
      </div></div>`,
      i.optimization ? steps(i.optimization) : "",
    ].join("");

    container.innerHTML = [
      section(num(), "Core concept", overview, "concept"),
      section(num(), "Mathematics &amp; mechanism", mathBody, "math"),
      section(num(), "Loss function &amp; optimization", lossBody, "loss"),
      i.assumptions ? section(num(), "Key assumptions",
        table(["Assumption", "Why it matters", "How to check"], i.assumptions.map((a) => [a.name, a.why, a.check])), "assumptions") : "",
      i.regularization ? section(num(), "Regularization &amp; variants",
        table(["Variant", "Formula", "Effect"], i.regularization.map((r) => [r.name, `<span class="inline-formula">${r.formula}</span>`, r.note])), "regularization") : "",
      section(num(), "Hyperparameters &amp; tuning",
        i.hyperparameters
          ? table(["Hyperparameter", "Typical range", "Effect of increasing", "Tuning strategy"],
              i.hyperparameters.map((h) => [h.name, `<span class="inline-formula">${h.range}</span>`, h.increasing, h.strategy]))
          : table(["Parameter", "Effect"], i.parameters.map((p) => [p.name, p.effect])), "hyperparameters"),
      section(num(), "Evaluation &amp; diagnostics", `
        <div class="info-grid">
          <div class="info-block"><h4>Evaluation metrics</h4>${list(i.metrics)}</div>
          <div class="info-block"><h4>Typical uses</h4>${list(i.typicalUses)}</div>
          ${i.diagnostics ? `<div class="info-block span2"><h4>Diagnostics</h4>${list(i.diagnostics)}</div>` : ""}
        </div>`, "evaluation"),
      (i.advantages || i.limitations) ? section(num(), "Advantages &amp; limitations", `
        <div class="pro-con">
          ${i.advantages ? `<div class="info-block pros"><h4>Advantages</h4>${list(i.advantages)}</div>` : ""}
          ${i.limitations ? `<div class="info-block cons"><h4>Limitations</h4>${list(i.limitations.map((l) => typeof l === "string" ? l : `${l.name} - ${l.note}${l.fix ? ` <i>Fix: ${l.fix}</i>` : ""}`))}</div>` : ""}
        </div>
        ${i.alternatives ? table(["Reach for instead", "When"], i.alternatives.map((a) => [a.name, a.when])) : ""}`, "proscons") : "",
      i.pitfalls ? section(num(), "Common pitfalls",
        table(["Pitfall", "Solution"], i.pitfalls.map((p) => [p.problem, p.solution])), "pitfalls") : "",
      i.workedExample ? section(num(), "Worked example - by hand", `
        <div class="info-grid"><div class="info-block span2 worked-example">
          <p><b>${i.workedExample.setup}</b></p>
          <ol class="info-list">${i.workedExample.steps.map((s) => `<li>${s}</li>`).join("")}</ol>
          <p class="formula">${i.workedExample.result}</p>
        </div></div>`, "worked") : "",
      (i.quickRef || i.code) ? section(num(), "Quick reference", [
        i.quickRef ? table(["Component", "Formula"], i.quickRef.map((q) => [q.name, `<span class="inline-formula">${q.formula}</span>`])) : "",
        codeBlock(i.code),
      ].join(""), "quickref") : "",
      i.whyChain ? section(num(), "Why-chain - interview drill", whyChain(i.whyChain), "whychain") : "",
    ].join("");

    if (i.decisionFunction.plot) MLU.plotFn(container.querySelector("#plot-decision"), i.decisionFunction.plot);
    if (i.lossFunction.plot) MLU.plotFn(container.querySelector("#plot-loss"), i.lossFunction.plot);
  }

  function router() {
    const sidebar = document.getElementById("sidebar-nav");
    if (sidebar) sidebar.classList.remove("open"); // collapse the mobile drawer on navigation
    const hash = location.hash.replace(/^#\/?/, "");
    if (!hash) renderHome();
    else renderAlgo(hash);
    // Land at the top of the new page rather than inheriting the previous
    // route's offset. `.main` is the scroll container on desktop; the window
    // scrolls instead once the layout stacks on mobile.
    const main = document.getElementById("main");
    if (main) main.scrollTop = 0;
    window.scrollTo(0, 0);
  }

  function currentTheme() {
    const stored = localStorage.getItem("ml-theme");
    if (stored) return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function updateThemeToggleUI(theme) {
    const moon = document.getElementById("theme-icon-moon");
    const sun = document.getElementById("theme-icon-sun");
    const label = document.getElementById("theme-toggle-label");
    if (moon) moon.style.display = theme === "dark" ? "" : "none";
    if (sun) sun.style.display = theme === "dark" ? "none" : "";
    if (label) label.textContent = theme;
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ml-theme", theme);
    updateThemeToggleUI(theme);
  }

  function initTheme() {
    updateThemeToggleUI(currentTheme());
    const stored = localStorage.getItem("ml-theme");
    if (stored) document.documentElement.setAttribute("data-theme", stored);
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.addEventListener("click", () => applyTheme(currentTheme() === "dark" ? "light" : "dark"));
  }

  // ---- header search --------------------------------------------------------
  // Two kinds of hit: the algorithm itself (name/tagline/category), and any
  // passage inside its reference page, so a concept or interview question can be
  // found without knowing which model it belongs to.

  // Where each indexed passage lives, mapped to the section slug it scrolls to.
  const SECTIONS = {
    concept:        { label: "Core concept",    slug: "concept" },
    math:           { label: "Mathematics",     slug: "math" },
    loss:           { label: "Loss & optim.",   slug: "loss" },
    assumptions:    { label: "Assumptions",     slug: "assumptions" },
    regularization: { label: "Regularization",  slug: "regularization" },
    hyperparameters:{ label: "Hyperparameters", slug: "hyperparameters" },
    evaluation:     { label: "Evaluation",      slug: "evaluation" },
    proscons:       { label: "Pros & cons",     slug: "proscons" },
    pitfalls:       { label: "Pitfalls",        slug: "pitfalls" },
    worked:         { label: "Worked example",  slug: "worked" },
    quickref:       { label: "Quick reference", slug: "quickref" },
    whychain:       { label: "Interview Q",     slug: "whychain" },
  };

  const strip = (s) => String(s).replace(/<[^>]*>/g, "");

  // Badge text for a hit. Algorithm pages use the fixed SECTIONS table; guide
  // pages label hits with their own section title.
  function sectionLabel(algo, slug) {
    if (algo.kind === "guide") {
      const sec = (algo.guide || []).find((s) => s.slug === slug);
      if (sec) return sec.title.replace(/&amp;/g, "&");
    }
    return SECTIONS[slug] ? SECTIONS[slug].label : slug;
  }

  // Flattens one algorithm's info into { section, text } passages. Built once,
  // lazily, because every algorithm file must have registered first.
  function buildEntries(a) {
    const i = a.info || {};
    const out = [];
    const push = (sec, text) => { if (text) out.push({ section: sec, text: strip(text) }); };
    const pushAll = (sec, arr, fn) => (arr || []).forEach((x) => push(sec, fn(x)));

    // Guide pages carry typed blocks instead of the algorithm schema; index each
    // block against its own section so hits scroll to the right place.
    if (a.kind === "guide") {
      for (const sec of a.guide || []) {
        const s = sec.slug;
        for (const b of sec.body || []) {
          if (typeof b === "string") { push(s, b); continue; }
          if (b.p) push(s, b.p);
          if (b.list) b.list.forEach((x) => push(s, x));
          if (b.olist) b.olist.forEach((x) => push(s, x));
          if (b.table) b.table.rows.forEach((r) => push(s, r.join(" - ")));
          if (b.steps) b.steps.forEach((x) => push(s, `${x.title}: ${x.formula || ""} ${x.note || ""}`));
          if (b.flow) b.flow.forEach((x) => push(s, `${x.label} ${x.note || ""}`));
          if (b.lanes) b.lanes.forEach((l) => { push(s, `${l.name}: ${l.note || ""}`); (l.items || []).forEach((it) => push(s, it)); });
          if (b.branch) { push(s, b.branch.q); (b.branch.arms || []).forEach((x) => push(s, `${x.when}: ${x.then} ${x.note || ""}`)); }
          if (b.callout) push(s, `${b.callout.title}: ${b.callout.text}`);
          if (b.qa) b.qa.forEach((qa) => { push(s, qa.q); push(s, `${qa.q} ${qa.a}`); });
          if (b.cards) b.cards.forEach((c) => push(s, `${c.name}: ${c.note || (c.list || []).join(" ")}`));
          if (b.code) push(s, b.code);
        }
      }
      // Attribute the page summary to the first section, since guides have no
      // fixed "concept" anchor to scroll to.
      const first = (a.guide || [])[0];
      if (first) push(first.slug, a.description);
      return out;
    }

    if (i.intuition) {
      push("concept", i.intuition.definition);
      pushAll("concept", i.intuition.steps, (s) => s);
      pushAll("concept", i.intuition.applications, (s) => s);
    }
    push("concept", i.type); push("concept", i.scenario);
    push("concept", i.inputs); push("concept", i.output);

    pushAll("math", i.math, (m) => `${m.title}: ${m.formula || ""} ${m.note || ""}`);
    pushAll("math", i.pipeline, (p) => `${p.label} ${p.note || ""}`);
    if (i.decisionFunction) push("math", `${i.decisionFunction.text} ${i.decisionFunction.mechanism}`);
    if (i.lossFunction) push("loss", `${i.lossFunction.text} ${i.lossFunction.mechanism}`);
    pushAll("loss", i.optimization, (o) => `${o.title}: ${o.formula || ""} ${o.note || ""}`);

    pushAll("assumptions", i.assumptions, (x) => `${x.name}: ${x.why} ${x.check}`);
    pushAll("regularization", i.regularization, (x) => `${x.name}: ${x.formula} ${x.note}`);
    pushAll("hyperparameters", i.hyperparameters, (h) => `${h.name} (${h.range}): ${h.increasing} ${h.strategy}`);
    pushAll("hyperparameters", i.parameters, (p) => `${p.name}: ${p.effect}`);

    pushAll("evaluation", i.metrics, (s) => s);
    pushAll("evaluation", i.typicalUses, (s) => s);
    pushAll("evaluation", i.diagnostics, (s) => s);

    pushAll("proscons", i.advantages, (s) => s);
    pushAll("proscons", i.limitations, (l) => typeof l === "string" ? l : `${l.name}: ${l.note} ${l.fix || ""}`);
    pushAll("proscons", i.alternatives, (x) => `${x.name}: ${x.when}`);

    pushAll("pitfalls", i.pitfalls, (p) => `${p.problem} -> ${p.solution}`);
    pushAll("quickref", i.quickRef, (q) => `${q.name}: ${q.formula}`);
    push("quickref", i.code);

    if (i.workedExample) {
      push("worked", i.workedExample.setup);
      pushAll("worked", i.workedExample.steps, (s) => s);
    }
    // Indexed as question and answer separately so a question-shaped query
    // matches the question text directly.
    (i.whyChain || []).forEach((qa) => { push("whychain", qa.q); push("whychain", `${qa.q} ${qa.a}`); });

    return out;
  }

  let searchIndex = null;
  function getIndex() {
    if (!searchIndex) {
      searchIndex = registry.map((a) => ({ algo: a, entries: buildEntries(a) }));
    }
    return searchIndex;
  }

  // Dropped before matching so a typed-out question ("why does L1 produce
  // sparse weights") is judged on its content words, not its grammar.
  const STOPWORDS = new Set(["a","an","the","is","are","was","were","be","do","does","did","of","to","in","on","for","and","or","with","that","this","it","its","as","at","by","from","how","what","why","when","which","who","i","you","we","my","me","can","could","should","would","if","not","no","vs","versus"]);

  function queryTerms(q) {
    const raw = q.split(/\s+/).filter(Boolean);
    const kept = raw.filter((t) => !STOPWORDS.has(t) && t.length > 1);
    return kept.length ? kept : raw;   // an all-stopword query still searches
  }

  // How many of the terms appear in the text.
  function termHits(text, terms) {
    let n = 0;
    for (const t of terms) if (text.includes(t)) n++;
    return n;
  }

  // Rarity weight per term, so a distinctive word ("l1", "hessian") outranks a
  // ubiquitous one ("sparse", "model") when results have to be relaxed.
  function termWeights(terms) {
    const idx = getIndex();
    const w = new Map();
    for (const t of terms) {
      let df = 0;
      for (const { algo, entries } of idx) {
        const inAlgo = algo.name.toLowerCase().includes(t) || entries.some((e) => e.text.toLowerCase().includes(t));
        if (inAlgo) df++;
      }
      w.set(t, 1 / Math.log(1 + Math.max(df, 1)));
    }
    return w;
  }

  function weightedScore(text, terms, weights) {
    let s = 0;
    for (const t of terms) if (text.includes(t)) s += weights.get(t) || 0;
    return s;
  }

  // Returns [{ algo, score, hits: [{section, text}] }], best algorithm first.
  // Long natural-language queries rarely match every word, so a majority is
  // enough; if even that finds nothing, relax until something surfaces rather
  // than dead-ending on a phrasing the notes happen to word differently.
  function searchAll(query) {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    const terms = queryTerms(q);
    const strict = terms.length <= 2 ? terms.length : Math.ceil(terms.length * 0.6);
    const weights = termWeights(terms);
    for (let need = strict; need >= 1; need--) {
      const found = searchPass(q, terms, need, weights);
      if (found.length) return found;
    }
    return [];
  }

  function searchPass(q, terms, need, weights) {
    const results = [];

    for (const { algo, entries } of getIndex()) {
      const name = algo.name.toLowerCase();
      const meta = `${name} ${algo.tagline} ${algo.category} ${algo.description}`.toLowerCase();

      // Algorithm-level relevance: a name prefix beats a name substring, which
      // beats a mention anywhere in the summary text.
      let score = 100;
      if (name.startsWith(q)) score = 0;
      else if (name.includes(q)) score = 1;
      else if (termHits(meta, terms) === terms.length) score = 2;

      // Content-level relevance: term coverage, weighted so rare terms dominate.
      const scoredHits = [];
      for (const e of entries) {
        const low = e.text.toLowerCase();
        const n = termHits(low, terms);
        if (n < need) continue;
        // Prefer a passage containing the query verbatim.
        const exact = low.includes(q) ? 1 : 0;
        scoredHits.push({ entry: e, n, exact, w: weightedScore(low, terms, weights) });
      }
      scoredHits.sort((x, y) => y.exact - x.exact || y.w - x.w || y.n - x.n);
      const hits = scoredHits.slice(0, 4).map((h) => h.entry);
      const best = scoredHits.length ? scoredHits[0].w : 0;

      if (score === 100 && !hits.length) continue;
      // An algorithm with content hits but no name match still ranks below
      // direct name matches, and above nothing.
      if (score === 100) score = 3;
      results.push({ algo, score, hits, hitCount: hits.length, best });
    }

    return results
      .sort((x, y) => x.score - y.score || y.best - x.best || y.hitCount - x.hitCount
                      || x.algo.name.localeCompare(y.algo.name))
      .slice(0, 6);
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Marks every query term, so a multi-word query lights up each word it hit.
  function highlight(text, query) {
    const safe = escapeHtml(text);
    const terms = queryTerms(query.trim().toLowerCase())
      .filter(Boolean)
      .sort((a, b) => b.length - a.length)          // longest first, so nesting cannot occur
      .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (!terms.length) return safe;
    return safe.replace(new RegExp(`(${terms.join("|")})`, "gi"), "<mark>$1</mark>");
  }

  // Trims a long passage to a window around the earliest matching term, so the
  // reason the result matched is actually visible in the snippet.
  function snippet(text, query, width = 130) {
    const low = text.toLowerCase();
    const q = query.trim().toLowerCase();
    let i = low.indexOf(q);                          // whole phrase if present
    if (i === -1) {
      for (const t of queryTerms(q)) {
        const at = low.indexOf(t);
        if (at !== -1 && (i === -1 || at < i)) i = at;
      }
    }
    if (i === -1) {
      return highlight(text.slice(0, width) + (text.length > width ? "…" : ""), query);
    }
    const start = Math.max(0, i - Math.floor(width / 3));
    const end = Math.min(text.length, start + width);
    const cut = (start > 0 ? "…" : "") + text.slice(start, end) + (end < text.length ? "…" : "");
    return highlight(cut, query);
  }

  function initSearch() {
    const box = document.getElementById("search");
    const input = document.getElementById("search-input");
    const panel = document.getElementById("search-results");
    if (!box || !input || !panel) return;

    // Flat list of selectable rows, in DOM order, so the arrow keys can walk
    // across group boundaries. Each row is { algo, section? }.
    let rows = [];
    let cursor = -1;

    function close() {
      panel.hidden = true;
      input.setAttribute("aria-expanded", "false");
      cursor = -1;
    }

    function go(row) {
      if (!row) return;
      pendingFocus = row.section || null;
      const sameAlgo = active === row.algo.id;
      location.hash = "#/" + row.algo.id;
      // hashchange does not fire when the hash is unchanged, so re-route by hand.
      if (sameAlgo) router();
      input.value = "";
      close();
      input.blur();
    }

    function paint() {
      const els = panel.querySelectorAll(".search-item");
      els.forEach((el, i) => el.classList.toggle("active", i === cursor));
      if (cursor >= 0 && els[cursor]) els[cursor].scrollIntoView({ block: "nearest" });
    }

    function render() {
      const q = input.value;
      cursor = -1;
      rows = [];
      if (q.trim().length < 2) { close(); return; }

      const results = searchAll(q);

      if (!results.length) {
        panel.innerHTML = `<div class="search-empty">Nothing matches “${escapeHtml(q)}”</div>`;
      } else {
        panel.innerHTML = results.map((r) => {
          const meta = CATEGORY_META[r.algo.category] || { color: "var(--text)", icon: "" };
          // The group header itself is a row: it opens the algorithm page.
          rows.push({ algo: r.algo });
          const head = `<div class="search-group-head">
              <span class="search-item-icon">${meta.icon}</span>
              <span class="search-group-name">${highlight(r.algo.name, q)}</span>
              <span class="search-group-cat">${escapeHtml(r.algo.category)}</span>
            </div>`;
          const openRow = `<div class="search-item search-item-open" role="option">
              <span class="search-badge">Open</span>
              <span class="search-item-text"><span class="search-item-meta">${escapeHtml(r.algo.tagline)}</span></span>
            </div>`;
          const hitRows = r.hits.map((h) => {
            rows.push({ algo: r.algo, section: h.section });
            return `<div class="search-item" role="option">
              <span class="search-badge">${escapeHtml(sectionLabel(r.algo, h.section))}</span>
              <span class="search-item-text"><span class="search-item-snippet">${snippet(h.text, q)}</span></span>
            </div>`;
          }).join("");
          return `<div class="search-group" style="--cat-color:${meta.color}">${head}${openRow}${hitRows}</div>`;
        }).join("");

        panel.querySelectorAll(".search-item").forEach((el, i) => {
          el.onmousedown = (e) => { e.preventDefault(); go(rows[i]); };
          el.onmouseenter = () => { cursor = i; paint(); };
        });
        panel.querySelectorAll(".search-group-head").forEach((el) => {
          el.onmousedown = (e) => {
            e.preventDefault();
            const g = el.parentElement;
            const idx = [...panel.querySelectorAll(".search-item")].indexOf(g.querySelector(".search-item"));
            go(rows[idx]);
          };
        });
      }
      panel.hidden = false;
      input.setAttribute("aria-expanded", "true");
    }

    input.addEventListener("input", render);
    input.addEventListener("focus", () => { if (input.value.trim()) render(); });
    input.addEventListener("blur", () => setTimeout(close, 120));

    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { input.value = ""; close(); input.blur(); return; }
      if (panel.hidden || !rows.length) return;
      if (e.key === "ArrowDown") { e.preventDefault(); cursor = (cursor + 1) % rows.length; paint(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); cursor = (cursor - 1 + rows.length) % rows.length; paint(); }
      else if (e.key === "Enter") { e.preventDefault(); go(rows[cursor >= 0 ? cursor : 0]); }
    });

    // "/" focuses the box from anywhere, as long as you are not already typing.
    document.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (e.key === "/" && tag !== "input" && tag !== "textarea" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        input.focus();
      }
    });
  }

  function initMobileNav() {
    const btn = document.getElementById("menu-toggle");
    const sidebar = document.getElementById("sidebar-nav");
    if (btn && sidebar) btn.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  function init() {
    initTheme();
    initSearch();
    initMobileNav();
    window.addEventListener("hashchange", router);
    router();
  }

  // `registry` and `renderInfo` are exposed so the reference pages can be
  // rendered headlessly and checked for gaps without a browser.
  return { register, init, registry, renderInfo, renderGuideBody, searchAll };
})();
