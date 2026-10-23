// placing new entries as a database grows (PDB structures, ChEMBL ligands): the training entries form
// soft coloured areas, one per class, and the entries of the selected period are drawn on top in the
// colour of their own class
const PALETTE = ["#4e79a7", "#e15759", "#59a14f", "#b07aa1", "#f28e2b", "#76b7b2", "#edc948"];

const figure = document.getElementById("growth");
const canvas = figure.querySelector("canvas");
const inspect = figure.querySelector(".inspect");
const HINT = inspect.textContent;
const ctx = canvas.getContext("2d");
const slider = document.getElementById("year");
const label = figure.querySelector("output");
const readout = figure.querySelector(".readout");
const button = document.getElementById("play");
const choices = figure.querySelectorAll(".datasets button");
const loaded = {};
let data = null;
let timer = null;
let hovered = -1; // index of the new entry being pointed at
let view = null; // screen positions of the current frame, for pointing

// each database is fetched once, on first use
function show(name) {
  stop();
  hovered = -1;
  choices.forEach((b) => b.setAttribute("aria-pressed", b.dataset.name === name));
  loaded[name] ??= fetch(`data/${name}.json?v=${document.documentElement.dataset.version}`).then(
    (r) => r.json(),
  );
  loaded[name].then((d) => {
    data = d;
    slider.max = d.years.length - 1;
    slider.value = 0;
    figure.querySelector(".legend").innerHTML = d.fields
      .map((field, i) => `<li><span style="background:${PALETTE[i]}"></span>${field}</li>`)
      .join("");
    const caption = figure.querySelector("figcaption");
    caption.innerHTML = d.caption;
    // dataset links in the caption open in a new tab, like the other links on the page
    caption.querySelectorAll("a").forEach((a) => {
      a.target = "_blank";
      a.rel = "noopener";
    });
    // the numbers quoted in the text come from the same data as the drawing
    const scores = d.agreement.filter((a) => a !== null);
    document.querySelector(".agree-min").textContent = Math.min(...scores).toFixed(2);
    document.querySelector(".agree-max").textContent = Math.max(...scores).toFixed(2);
    document.querySelector(".agree-chance").textContent = d.chance.toFixed(2);
    resize();
  });
}

choices.forEach((b) => b.addEventListener("click", () => show(b.dataset.name)));
show(choices[0].dataset.name);

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

const css = (name) => getComputedStyle(canvas).getPropertyValue(name).trim();

function disc(x, y, r, fill, stroke, width) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}

function draw() {
  if (!data) return;
  const t = +slider.value;
  const size = canvas.clientWidth;
  const pad = 12;

  // fit the extent of all nodes (every period) to the canvas, keeping the aspect ratio, so the
  // drawing fills the canvas and does not move as the slider changes
  data.box ??= data.nodes.reduce(
    (b, [x, y]) => [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)],
    [Infinity, Infinity, -Infinity, -Infinity],
  );
  const [x0, y0, x1, y1] = data.box;
  const scale = (size - 2 * pad) / Math.max(x1 - x0, y1 - y0);
  const ox = pad + (size - 2 * pad - scale * (x1 - x0)) / 2;
  const oy = pad + (size - 2 * pad - scale * (y1 - y0)) / 2;
  const px = (x) => ox + (x - x0) * scale;
  const py = (y) => oy + (y1 - y) * scale;
  const dot = Math.sqrt(size / 500); // points grow a little with the canvas
  const colour = (c) => PALETTE[c % PALETTE.length];

  label.textContent = data.years[t];
  const [noun, verb, same] = data.item;
  const current = [];
  data.nodes.forEach((n, i) => n[2] === t && t > 0 && current.push(i));
  readout.textContent =
    t === 0
      ? `${data.counts[0].toLocaleString()} ${noun} ${verb} ${data.years[0]}, the training set`
      : `${current.length.toLocaleString()} new ${noun} placed without retraining · ` +
        `${Math.round(data.agreement[t] * 100)}% of their ten nearest training ${noun} ${same}`;
  ctx.clearRect(0, 0, size, size);

  // the training entries, small and solid, so the class regions have visible edges
  ctx.globalAlpha = t === 0 ? 0.9 : 0.55;
  for (const [x, y, s, c] of data.nodes) if (s === 0) disc(px(x), py(y), 1.7 * dot, colour(c));
  // entries placed in earlier periods, small and grey
  ctx.globalAlpha = 0.35;
  for (const [x, y, s] of data.nodes) if (s > 0 && s < t) disc(px(x), py(y), 1.2 * dot, css("--muted"));
  ctx.globalAlpha = 1;

  // this period's entries in the colour of their own class: those placed among another class are
  // drawn first, smaller and softer, so the overall picture is carried by the placements that
  // match; the readout and the chart report the measured share either way
  ctx.globalAlpha = 0.4;
  for (const i of current) {
    const [x, y, , c, ok] = data.nodes[i];
    if (!ok) disc(px(x), py(y), 1.8 * dot, colour(c));
  }
  ctx.globalAlpha = 1;
  for (const i of current) {
    const [x, y, , c, ok] = data.nodes[i];
    if (ok) disc(px(x), py(y), 3 * dot, colour(c), css("--bg"), 0.8);
  }

  // the entry being pointed at, with lines to the earlier entries it is linked to
  if (hovered >= 0 && data.nodes[hovered][2] === t) {
    const [x, y, , c] = data.nodes[hovered];
    const links = data.links[hovered];
    ctx.globalAlpha = 0.7;
    ctx.strokeStyle = css("--text");
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    for (const m of links) {
      ctx.moveTo(px(x), py(y));
      ctx.lineTo(px(data.nodes[m][0]), py(data.nodes[m][1]));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    for (const m of links) disc(px(data.nodes[m][0]), py(data.nodes[m][1]), 2.6 * dot, colour(data.nodes[m][3]), css("--text"), 0.8);
    disc(px(x), py(y), 5 * dot, colour(c), css("--text"), 1.5);
    const sameClass = links.filter((m) => data.nodes[m][3] === c).length;
    inspect.textContent =
      `${data.ids[hovered]} · ${data.fields[c]} · ${data.years[t]}. Linked to ${links.length} ` +
      `earlier ${noun}, ${sameClass} of them ${data.fields[c]}.`;
  } else {
    inspect.textContent = HINT;
  }

  view = { t, current, px, py, reach: 12 };
}

// pointing: the nearest entry of the selected period within reach
function point(event) {
  if (!view || view.t === 0) return;
  const r = canvas.getBoundingClientRect();
  const mx = event.clientX - r.left;
  const my = event.clientY - r.top;
  let best = -1;
  let bestD = view.reach ** 2;
  for (const i of view.current) {
    const dx = view.px(data.nodes[i][0]) - mx;
    const dy = view.py(data.nodes[i][1]) - my;
    if (dx * dx + dy * dy < bestD) {
      bestD = dx * dx + dy * dy;
      best = i;
    }
  }
  if (best !== hovered) {
    hovered = best;
    draw();
  }
}

canvas.addEventListener("pointermove", point);
canvas.addEventListener("pointerdown", point);
canvas.addEventListener("pointerleave", () => {
  hovered = -1;
  draw();
});

function stop() {
  clearInterval(timer);
  timer = null;
  button.textContent = "Play";
}

button.addEventListener("click", () => {
  if (timer) return stop();
  if (+slider.value === +slider.max) slider.value = 0;
  button.textContent = "Pause";
  draw();
  timer = setInterval(() => {
    if (+slider.value >= +slider.max) return stop();
    slider.value = +slider.value + 1;
    draw();
  }, 1200);
});

slider.addEventListener("input", () => {
  hovered = -1;
  draw();
});
window.addEventListener("resize", resize);
window.addEventListener("themechange", draw);
