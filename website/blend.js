// FlingBlend slider: one fitted model, drawn at 41 values of lambda
(() => {
  const canvas = document.querySelector("#blend canvas");
  const ctx = canvas.getContext("2d");
  const slider = document.getElementById("lam");
  const label = document.querySelector("#blend output");
  const readout = document.querySelector("#blend .readout");
  let data = null;

  fetch(`data/blend.json?v=${document.documentElement.dataset.version}`)
    .then((r) => r.json())
    .then((d) => {
      data = d;
      slider.max = d.lams.length - 1;
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
    if (!data) return;
    const k = +slider.value;
    const lam = data.lams[k];
    const pos = data.pos[k];
    const size = canvas.clientWidth;
    const pad = 12;

    // fit the extent over every value of lambda, so the frame stays fixed while sliding
    data.box ??= data.pos.flat().reduce(
      (b, [x, y]) => [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)],
      [Infinity, Infinity, -Infinity, -Infinity],
    );
    const [x0, y0, x1, y1] = data.box;
    const scale = (size - 2 * pad) / Math.max(x1 - x0, y1 - y0);
    const ox = pad + (size - 2 * pad - scale * (x1 - x0)) / 2;
    const oy = pad + (size - 2 * pad - scale * (y1 - y0)) / 2;
    const px = (x) => ox + (x - x0) * scale;
    const py = (y) => oy + (y1 - y) * scale;
    const trained = data.trained.includes(lam);

    label.textContent = `λ = ${lam.toFixed(3)}`;
    readout.textContent =
      `stress ${data.stress[k].toFixed(3)} · NP ${data.np[k].toFixed(3)} · ` +
      (trained ? "trained at this λ" : "not seen in training");
    ctx.clearRect(0, 0, size, size);

    ctx.strokeStyle = getComputedStyle(canvas).getPropertyValue("--edge");
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (const [a, b] of data.edges) {
      ctx.moveTo(px(pos[a][0]), py(pos[a][1]));
      ctx.lineTo(px(pos[b][0]), py(pos[b][1]));
    }
    ctx.stroke();

    pos.forEach(([x, y], i) => {
      ctx.fillStyle = data.colours[i];
      ctx.beginPath();
      ctx.arc(px(x), py(y), 3 * Math.sqrt(size / 500), 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  slider.addEventListener("input", draw);
  window.addEventListener("resize", resize);
  window.addEventListener("themechange", draw);
})();
