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
  let active = null;
  let activeCleanup = null;

  function register(algo) { registry.push(algo); }

  function byId(id) { return registry.find((a) => a.id === id); }

  function renderSidebar() {
    const nav = document.getElementById("sidebar-nav");
    MLU.clearNode(nav);
    for (const cat of order) {
      const items = registry.filter((a) => a.category === cat);
      if (!items.length) continue;
      const group = document.createElement("div");
      group.className = "sidebar-group";
      const title = document.createElement("div");
      title.className = "sidebar-group-title";
      title.textContent = cat;
      group.appendChild(title);
      for (const algo of items) {
        const item = document.createElement("div");
        item.className = "nav-item" + (algo.id === active ? " active" : "");
        item.innerHTML = `<span class="name">${algo.name}</span><span class="tag">${algo.tagline}</span>`;
        item.onclick = () => { location.hash = "#/" + algo.id; };
        group.appendChild(item);
      }
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
      <div class="home-grid">
        ${registry.map((a) => `
          <div class="home-card" data-id="${a.id}">
            <div class="cat">${a.category}</div>
            <div class="name">${a.name}</div>
            <div class="desc">${a.description}</div>
          </div>`).join("")}
      </div>
    `;
    main.querySelectorAll(".home-card").forEach((el) => {
      el.onclick = () => { location.hash = "#/" + el.dataset.id; };
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

  function renderInfo(container, info) {
    const list = (arr) => `<ul class="info-list">${arr.map((x) => `<li>${x}</li>`).join("")}</ul>`;
    container.innerHTML = `
      <div class="info-grid">
        <div class="info-block"><h4>Type of algorithm</h4><p>${info.type}</p></div>
        <div class="info-block"><h4>When it's used</h4><p>${info.scenario}</p></div>
        <div class="info-block"><h4>Inputs</h4><p>${info.inputs}</p></div>
        <div class="info-block"><h4>Output</h4><p>${info.output}</p></div>
        <div class="info-block span2">
          <h4>Decision function &amp; mechanism</h4>
          <p class="formula">${info.decisionFunction.text}</p>
          <p>${info.decisionFunction.mechanism}</p>
          ${info.decisionFunction.plot ? `<div class="plot-slot" id="plot-decision"></div><div class="plot-caption">${info.decisionFunction.plot.caption || ""}</div>` : ""}
        </div>
        <div class="info-block span2">
          <h4>Loss function &amp; mechanism</h4>
          <p class="formula">${info.lossFunction.text}</p>
          <p>${info.lossFunction.mechanism}</p>
          ${info.lossFunction.plot ? `<div class="plot-slot" id="plot-loss"></div><div class="plot-caption">${info.lossFunction.plot.caption || ""}</div>` : ""}
        </div>
        <div class="info-block span2">
          <h4>Parameters &amp; tuning</h4>
          <table class="param-table"><tbody>
            ${info.parameters.map((p) => `<tr><td>${p.name}</td><td>${p.effect}</td></tr>`).join("")}
          </tbody></table>
        </div>
        <div class="info-block"><h4>Evaluation metrics</h4>${list(info.metrics)}</div>
        <div class="info-block"><h4>Typical uses</h4>${list(info.typicalUses)}</div>
        ${info.workedExample ? `
        <div class="info-block span2 worked-example">
          <h4>Worked example — by hand</h4>
          <p><b>${info.workedExample.setup}</b></p>
          <ol class="info-list">${info.workedExample.steps.map((s) => `<li>${s}</li>`).join("")}</ol>
          <p class="formula">${info.workedExample.result}</p>
        </div>` : ""}
      </div>
    `;
    if (info.decisionFunction.plot) MLU.plotFn(container.querySelector("#plot-decision"), info.decisionFunction.plot);
    if (info.lossFunction.plot) MLU.plotFn(container.querySelector("#plot-loss"), info.lossFunction.plot);
  }

  function router() {
    const hash = location.hash.replace(/^#\/?/, "");
    if (!hash) renderHome();
    else renderAlgo(hash);
  }

  function init() {
    window.addEventListener("hashchange", router);
    router();
  }

  return { register, init };
})();
