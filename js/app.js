// App shell: registry, sidebar navigation, hash router.
// Each algorithm file calls MLApp.register({...}) when it loads.
const MLApp = (() => {
  const registry = [];
  const order = [
    "Supervised — Regression",
    "Supervised — Classification",
    "Supervised — Trees & Ensembles",
    "Unsupervised — Clustering",
    "Unsupervised — Dimensionality Reduction",
    "Deep Learning",
    "Reinforcement Learning",
  ];
  const CATEGORY_META = {
    "Supervised — Regression": {
      color: "var(--pastel-green)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17 L9 11 L13 14 L20 5"/><circle cx="4" cy="17" r="1.3" fill="currentColor" stroke="none"/><circle cx="9" cy="11" r="1.3" fill="currentColor" stroke="none"/><circle cx="13" cy="14" r="1.3" fill="currentColor" stroke="none"/><circle cx="20" cy="5" r="1.3" fill="currentColor" stroke="none"/></svg>`,
    },
    "Supervised — Classification": {
      color: "var(--pastel-red)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="21" x2="21" y2="3"/><circle cx="7" cy="8" r="1.5" fill="currentColor" stroke="none"/><circle cx="11" cy="6" r="1.5" fill="currentColor" stroke="none"/><circle cx="9" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="15" cy="18" r="1.5"/><circle cx="18" cy="14" r="1.5"/><circle cx="14" cy="20" r="1.5"/></svg>`,
    },
    "Supervised — Trees & Ensembles": {
      color: "var(--series-4)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="4" r="1.6" fill="currentColor" stroke="none"/><line x1="12" y1="5.6" x2="12" y2="9"/><line x1="12" y1="9" x2="6" y2="14"/><line x1="12" y1="9" x2="18" y2="14"/><circle cx="6" cy="15.6" r="1.5"/><circle cx="18" cy="15.6" r="1.5"/><line x1="6" y1="17" x2="4" y2="21"/><line x1="6" y1="17" x2="8" y2="21"/><line x1="18" y1="17" x2="16" y2="21"/><line x1="18" y1="17" x2="20" y2="21"/></svg>`,
    },
    "Unsupervised — Clustering": {
      color: "var(--series-1)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="9" r="5"/><circle cx="16" cy="14" r="4"/><circle cx="7" cy="17" r="3"/></svg>`,
    },
    "Unsupervised — Dimensionality Reduction": {
      color: "var(--series-5)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="20" x2="21" y2="20"/><circle cx="6" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="12" cy="4" r="1.3" fill="currentColor" stroke="none"/><circle cx="17" cy="9" r="1.3" fill="currentColor" stroke="none"/><line x1="6" y1="6" x2="6" y2="20" stroke-dasharray="2 2"/><line x1="12" y1="4" x2="12" y2="20" stroke-dasharray="2 2"/><line x1="17" y1="9" x2="17" y2="20" stroke-dasharray="2 2"/></svg>`,
    },
    "Deep Learning": {
      color: "var(--series-6)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="4" cy="7" r="1.5"/><circle cx="4" cy="17" r="1.5"/><circle cx="12" cy="4" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="20" r="1.5"/><circle cx="20" cy="9" r="1.5"/><circle cx="20" cy="16" r="1.5"/><line x1="4" y1="7" x2="12" y2="4"/><line x1="4" y1="7" x2="12" y2="12"/><line x1="4" y1="17" x2="12" y2="12"/><line x1="4" y1="17" x2="12" y2="20"/><line x1="12" y1="4" x2="20" y2="9"/><line x1="12" y1="12" x2="20" y2="9"/><line x1="12" y1="12" x2="20" y2="16"/><line x1="12" y1="20" x2="20" y2="16"/></svg>`,
    },
    "Reinforcement Learning": {
      color: "var(--series-7)",
      icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.2"/><rect x="14" y="3" width="7" height="7" rx="1.2"/><rect x="3" y="14" width="7" height="7" rx="1.2"/><rect x="14" y="14" width="7" height="7" rx="1.2" fill="currentColor" stroke="none" opacity="0.45"/><path d="M10 6.5 H14 M17.5 10 V14 M6.5 14 V17.5"/></svg>`,
    },
  };

  let active = null;
  let activeCleanup = null;

  function register(algo) { registry.push(algo); }

  function byId(id) { return registry.find((a) => a.id === id); }

  // Sidebar groups start collapsed so the rail stays readable; the group
  // holding the current algorithm is always forced open, and any group the
  // user opens by hand is remembered across navigations.
  const openGroups = (() => {
    try { return new Set(JSON.parse(localStorage.getItem("ml-open-groups") || "[]")); }
    catch (e) { return new Set(); }
  })();

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
      const isOpen = openGroups.has(cat) || cat === activeCat;

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
        if (openGroups.has(cat)) openGroups.delete(cat); else openGroups.add(cat);
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
        <h1>ML Algorithms — Interactive Lab</h1>
        <p>A visual, hands-on companion to <b>rushter/MLAlgorithms</b> — a from-scratch collection of ML algorithms in Python/numpy, cloned alongside this page. Every demo below reimplements the same algorithm in JavaScript so you can click, drag and step through it live. Pick one from the sidebar, or a card below.</p>
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

  function renderAlgo(id) {
    const algo = byId(id);
    if (!algo) { renderHome(); return; }
    if (typeof activeCleanup === "function") { try { activeCleanup(); } catch (e) {} }
    activeCleanup = null;
    active = id;
    const main = document.getElementById("main");
    main.innerHTML = `
      <div class="main-header">
        <h1>${algo.name}</h1>
        <p>${algo.description}</p>
        <span class="source-link">reference: ${algo.sourceFile}</span>
      </div>
      <div class="tabs">
        <div class="tab active" data-tab="play">Playground</div>
        <div class="tab" data-tab="ref">Reference</div>
      </div>
      <div id="tab-play"></div>
      <div id="tab-ref" style="display:none"></div>
    `;
    renderSidebar();
    const playRoot = document.getElementById("tab-play");
    const cleanup = algo.mount(playRoot, MLU);
    if (typeof cleanup === "function") activeCleanup = cleanup;
    if (algo.info) renderInfo(document.getElementById("tab-ref"), algo.info);

    main.querySelectorAll(".tab").forEach((tab) => {
      tab.onclick = () => {
        main.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById("tab-play").style.display = tab.dataset.tab === "play" ? "" : "none";
        document.getElementById("tab-ref").style.display = tab.dataset.tab === "ref" ? "" : "none";
      };
    });
  }

  // ---- reference-tab rendering helpers -------------------------------------
  const list = (arr) => `<ul class="info-list">${arr.map((x) => `<li>${x}</li>`).join("")}</ul>`;

  // A titled full-width section of the reference page. Returns "" when the
  // section has no content, so every part of the schema below is optional and
  // algorithms can be migrated to the expanded format one at a time.
  function section(num, title, body) {
    if (!body) return "";
    return `<section class="ref-section">
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

  // Examiner-style Q&A chain — the "why does that follow?" drill-down.
  function whyChain(items) {
    if (!items || !items.length) return "";
    return `<div class="why-chain">${items.map((qa) => `
      <div class="why-item">
        <div class="why-q">${qa.q}</div>
        <div class="why-a">${qa.a}</div>
      </div>`).join("")}</div>`;
  }

  function codeBlock(src) {
    if (!src) return "";
    const esc = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return `<pre class="code-block"><code>${esc}</code></pre>`;
  }

  function renderInfo(container, info) {
    const i = info;
    const num = (() => { let n = 0; return () => ++n; })();

    // 1 — Core concept: definition, intuition steps, real-world uses.
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
      section(num(), "Core concept", overview),
      section(num(), "Mathematics &amp; mechanism", mathBody),
      section(num(), "Loss function &amp; optimization", lossBody),
      i.assumptions ? section(num(), "Key assumptions",
        table(["Assumption", "Why it matters", "How to check"], i.assumptions.map((a) => [a.name, a.why, a.check]))) : "",
      i.regularization ? section(num(), "Regularization &amp; variants",
        table(["Variant", "Formula", "Effect"], i.regularization.map((r) => [r.name, `<span class="inline-formula">${r.formula}</span>`, r.note]))) : "",
      section(num(), "Hyperparameters &amp; tuning",
        i.hyperparameters
          ? table(["Hyperparameter", "Typical range", "Effect of increasing", "Tuning strategy"],
              i.hyperparameters.map((h) => [h.name, `<span class="inline-formula">${h.range}</span>`, h.increasing, h.strategy]))
          : table(["Parameter", "Effect"], i.parameters.map((p) => [p.name, p.effect]))),
      section(num(), "Evaluation &amp; diagnostics", `
        <div class="info-grid">
          <div class="info-block"><h4>Evaluation metrics</h4>${list(i.metrics)}</div>
          <div class="info-block"><h4>Typical uses</h4>${list(i.typicalUses)}</div>
          ${i.diagnostics ? `<div class="info-block span2"><h4>Diagnostics</h4>${list(i.diagnostics)}</div>` : ""}
        </div>`),
      (i.advantages || i.limitations) ? section(num(), "Advantages &amp; limitations", `
        <div class="pro-con">
          ${i.advantages ? `<div class="info-block pros"><h4>Advantages</h4>${list(i.advantages)}</div>` : ""}
          ${i.limitations ? `<div class="info-block cons"><h4>Limitations</h4>${list(i.limitations.map((l) => typeof l === "string" ? l : `${l.name} — ${l.note}${l.fix ? ` <i>Fix: ${l.fix}</i>` : ""}`))}</div>` : ""}
        </div>
        ${i.alternatives ? table(["Reach for instead", "When"], i.alternatives.map((a) => [a.name, a.when])) : ""}`) : "",
      i.pitfalls ? section(num(), "Common pitfalls",
        table(["Pitfall", "Solution"], i.pitfalls.map((p) => [p.problem, p.solution]))) : "",
      i.workedExample ? section(num(), "Worked example — by hand", `
        <div class="info-grid"><div class="info-block span2 worked-example">
          <p><b>${i.workedExample.setup}</b></p>
          <ol class="info-list">${i.workedExample.steps.map((s) => `<li>${s}</li>`).join("")}</ol>
          <p class="formula">${i.workedExample.result}</p>
        </div></div>`) : "",
      (i.quickRef || i.code) ? section(num(), "Quick reference", [
        i.quickRef ? table(["Component", "Formula"], i.quickRef.map((q) => [q.name, `<span class="inline-formula">${q.formula}</span>`])) : "",
        codeBlock(i.code),
      ].join("")) : "",
      i.whyChain ? section(num(), "Why-chain — interview drill", whyChain(i.whyChain)) : "",
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

  function initMobileNav() {
    const btn = document.getElementById("menu-toggle");
    const sidebar = document.getElementById("sidebar-nav");
    if (btn && sidebar) btn.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  function init() {
    initTheme();
    initMobileNav();
    window.addEventListener("hashchange", router);
    router();
  }

  return { register, init };
})();
