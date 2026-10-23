// fill in the stress and NP under each drawing of the Variants comparison, from the metrics the
// image script saved, and in each column highlight the best value of the measure that column
// (graph) is judged on
(() => {
  const matrix = document.querySelector(".variant-matrix6");
  if (!matrix) return;
  const measures = JSON.parse(matrix.dataset.measures);

  fetch(`img/variants/metrics.json?v=${document.documentElement.dataset.version}`)
    .then((r) => r.json())
    .then((metrics) => {
      for (const [graph, key] of Object.entries(measures)) {
        const row = metrics[graph];
        const cells = [...matrix.querySelectorAll(`figcaption[data-graph="${graph}"]`)];
        // the bold marks the best of the three Fling variants; the baselines are for reference
        const values = cells
          .filter((c) => !c.closest(".baseline"))
          .map((c) => row[c.dataset.variant][key]);
        const best = key === "stress" ? Math.min(...values) : Math.max(...values);
        const mark = (text, on) => (on ? `<strong>${text}</strong>` : text);

        for (const cell of cells) {
          const m = row[cell.dataset.variant];
          const ranked = !cell.closest(".baseline");
          cell.innerHTML =
            mark(`stress ${m.stress.toFixed(3)}`, ranked && key === "stress" && m.stress === best) +
            " · " +
            mark(`NP ${m.np.toFixed(3)}`, ranked && key === "np" && m.np === best);
        }
      }
    });
})();
