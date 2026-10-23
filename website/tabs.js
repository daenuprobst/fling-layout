// tab groups (installation, quick demos) and the Output / Code switch inside each demo
(() => {
  function select(group, name) {
    for (const tab of group.querySelectorAll(":scope > .tablist > [role=tab]")) {
      tab.setAttribute("aria-selected", tab.dataset.panel === name);
    }
    for (const panel of group.querySelectorAll(":scope > .panel")) {
      panel.hidden = panel.dataset.panel !== name;
    }
  }

  for (const group of document.querySelectorAll(".tabs")) {
    for (const tab of group.querySelectorAll(":scope > .tablist > [role=tab]")) {
      tab.addEventListener("click", () => select(group, tab.dataset.panel));
    }
  }

  // each demo shows the image made by scripts/build_demos.py and the code it ran, loaded from
  // the same file, so the two can't drift apart
  for (const demo of document.querySelectorAll(".demo")) {
    const name = demo.dataset.demo;
    demo.innerHTML = `
      <div class="tabs">
        <div class="tablist" role="tablist">
          <button role="tab" aria-selected="true" data-panel="output">Output</button>
          <button role="tab" aria-selected="false" data-panel="code">Code</button>
        </div>
        <div class="panel" data-panel="output">
          <img src="img/demos/${name}.png?v=${document.documentElement.dataset.version}" alt="Output of the ${name} demo" loading="lazy">
        </div>
        <div class="panel" data-panel="code" hidden>
          <pre><code class="language-python"></code></pre>
          <a href="demos/${name}.py" download>Download ${name}.py</a>
        </div>
      </div>`;

    const group = demo.querySelector(".tabs");
    for (const tab of group.querySelectorAll("[role=tab]")) {
      tab.addEventListener("click", () => select(group, tab.dataset.panel));
    }

    fetch(`demos/${name}.py?v=${document.documentElement.dataset.version}`)
      .then((r) => r.text())
      .then((src) => {
        const code = demo.querySelector("code");
        code.textContent = src;
        if (window.hljs) hljs.highlightElement(code);
      });
  }

  if (window.hljs) {
    document.querySelectorAll("code.language-bash").forEach((el) => hljs.highlightElement(el));
  }
})();
