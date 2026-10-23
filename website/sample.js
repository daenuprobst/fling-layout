// "train on a sample, place the rest": one drawing per training fraction, with the nodes in the
// training sample in blue and every node the model never saw during training in red
(() => {
  const fig = document.getElementById("sample");
  if (!fig) return;
  const canvas = fig.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const slider = document.getElementById("fraction");
  const label = fig.querySelector("output");
  const readout = fig.querySelector(".readout");
  const caption = fig.querySelector("figcaption");
  const choices = fig.querySelectorAll(".datasets button");
  const TRAIN = "#1f4e79";
  const PLACED = "#d1495b";
  const notes = {
    jagmesh8:
      "jagmesh8, a finite-element mesh with six holes (George, 1978), 1,141 nodes. Blue nodes are " +
      "the training sample, red nodes were never seen during training and are placed by the model.",
    minnesota:
      "The road network of Minnesota (Gleich, SuiteSparse), 2,640 intersections joined by roads. " +
      "The model sees only the network, no coordinates. Rotated onto the real intersection " +
      "coordinates, the drawing accounts for about half of their variance (Procrustes R\u00b2 " +
      "0.51), as much when trained on 2% of the intersections as on all of them; hop counts are " +
      "only a rough proxy for road length. Blue intersections are the training sample, red ones " +
      "are placed by the model.",
    crack:
      "crack, a finite-element mesh from SuiteSparse with 10,240 nodes, ten times the size of " +
      "jagmesh8. Blue nodes are the training sample, red nodes are placed by the model.",
  };
  let all = null;
  let name = "minnesota";

  fetch(`data/sample_fit.json?v=${document.documentElement.dataset.version}`)
    .then((r) => r.json())
    .then((d) => {
      all = d;
      resize();
    });

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function draw() {
    if (!all) return;
    const g = all[name];
    const k = +slider.value;
    const pos = g.pos[k];
    const count = g.counts[k];
    const size = canvas.clientWidth;
    const pad = 12;
    const px = (x) => pad + ((x + 1) / 2) * (size - 2 * pad);
    const py = (y) => pad + ((1 - y) / 2) * (size - 2 * pad);
    const dot = Math.sqrt(size / 500);

    // nodes in the training sample: a prefix of one fixed random order, so samples are nested
    g.inTrain ??= g.counts.map((c) => {
      const mask = new Uint8Array(g.n);
      g.order.slice(0, c).forEach((v) => (mask[v] = 1));
      return mask;
    });
    const train = g.inTrain[k];
    const full = g.fractions[k] === 1;

    label.textContent = `${Math.round(g.fractions[k] * 100)}%`;
    readout.textContent = full
      ? `trained on all ${g.n.toLocaleString()} nodes · stress ${g.stress[k].toFixed(4)}`
      : `trained on ${count.toLocaleString()} of ${g.n.toLocaleString()} nodes · ` +
        `${(g.n - count).toLocaleString()} placed without training · stress ${g.stress[k].toFixed(4)} ` +
        `(${g.stress[g.stress.length - 1].toFixed(4)} with all nodes)`;
    caption.textContent = notes[name];
    ctx.clearRect(0, 0, size, size);

    ctx.strokeStyle = getComputedStyle(canvas).getPropertyValue("--edge");
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (const [a, b] of g.edges) {
      ctx.moveTo(px(pos[a][0]), py(pos[a][1]));
      ctx.lineTo(px(pos[b][0]), py(pos[b][1]));
    }
    ctx.stroke();

    // placed nodes first, the training sample on top
    for (const pass of [0, 1]) {
      ctx.fillStyle = pass && !full ? TRAIN : PLACED;
      for (let v = 0; v < g.n; v++) {
        if ((full ? 0 : train[v]) !== pass) continue;
        ctx.beginPath();
        ctx.arc(px(pos[v][0]), py(pos[v][1]), (pass ? 2.6 : 1.8) * dot, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  choices.forEach((b) =>
    b.addEventListener("click", () => {
      name = b.dataset.name;
      choices.forEach((c) => c.setAttribute("aria-pressed", c === b));
      draw();
    }),
  );
  slider.addEventListener("input", draw);
  window.addEventListener("resize", resize);
  window.addEventListener("themechange", draw);
})();
