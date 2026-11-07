/* ==========================================================================
   Test region analysis: destructive testing (cores) only. Uses forms.js.
   ========================================================================== */

document.addEventListener("DOMContentLoaded", initDtOnly);

function initDtOnly() {
  const form = document.getElementById("dtonly-form");
  if (!form) return;

  const summary     = document.getElementById("dt-error-summary");
  const serverError = document.getElementById("dt-server-error");
  const analyseBtn  = document.getElementById("dt-analyse-btn");
  const modalEl     = document.getElementById("dtonlyModal");
  const modal       = bootstrap.Modal.getOrCreateInstance(modalEl);
  returnFocusOnClose(modalEl, analyseBtn);
  let last = null, draft = null;            // last analysis: { json, notes, rows }

  const cores = createDataTable({
    table: document.getElementById("d-cores-table"),
    idPrefix: "d-cores",
    minRows: 8,
    what: "cores",
    countEl: document.getElementById("d-cores-count"),
    statusEl: document.getElementById("d-cores-status"),
    errorsEl: document.getElementById("d-cores-errors"),
    notesEl: document.getElementById("d-cores-notes"),
    onEdit: () => { serverError.hidden = true; if (!summary.hidden) refresh(); draft?.queue(); },
  });
  cores.setColumns([CORE_COLS.d, CORE_COLS.h, CORE_COLS.fc]);
  document.getElementById("d-cores-add").addEventListener("click", () => cores.addBlankRow());
  document.getElementById("d-cores-clear").addEventListener("click", () => cores.clear());

  /* ---------- error summary (shared with the other tabs) ---------- */
  form.querySelector(".table-card").classList.add("no-kit");   // the table validates itself
  const kit = createFieldKit({ form, scope: form, summary });
  const problems = show => cores.validate(show).problems;
  const refresh = () => kit.renderSummary(problems(false));

  /* ---------- submit ---------- */
  form.addEventListener("submit", async e => {
    e.preventDefault();
    cores.submitted = true;
    serverError.hidden = true;
    const list = problems(true);
    kit.showSummary(list);
    if (list.length) return;

    const { complete: rows, notes } = cores.validate(false);
    const payload = { "diam[]": rows.map(r => r.d), "height[]": rows.map(r => r.h), "fc[]": rows.map(r => r.fc) };

    setButtonBusy(analyseBtn, true, "Run analysis", "Analysing…");
    try {
      const json = await postJson("/analyse_dt_only/", payload);
      last = { json, notes, rows };
      showResults();
    } catch (err) {
      showServerError(serverError, "The analysis could not be run.", err);
    } finally {
      setButtonBusy(analyseBtn, false, "Run analysis", "Analysing…");
    }
  });

  /* ---------- results ---------- */
  const kvRows = rows => rows.map(r => r.group
    ? `<tr class="kv-group"><th colspan="2">${r.group}</th></tr>`
    : `<tr><th scope="row">${r[0]}</th><td>${r[1]}${r[2] ? `<span class="kv-note${/governs/.test(r[2]) ? " kv-warn" : ""}">${r[2]}</span>` : ""}</td></tr>`).join("");

  function showResults() {
    const { json, notes } = last;
    const D = json.dt_only;

    document.getElementById("dt-sub").textContent = `${D.n} cores`;

    // the server already reports cores outside the 44-160 mm diameters, so skip the form's duplicate note
    const warnings = [...(json.warnings || []), ...notes.filter(n => !/Core diameter/.test(n))];
    const wBox = document.getElementById("dt-warnings");
    wBox.querySelector("ul").innerHTML = warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");
    wBox.hidden = !warnings.length;

    document.getElementById("dt-fck").textContent = fx(D.fck);
    document.getElementById("dt-summary").innerHTML = kvRows([
      ["Mean in-situ strength, f<sub>cm(n),is</sub>", `${fx(D.fcm)} MPa`],
      ["Standard deviation, s<sub>fc,is</sub>", `${fx(D.s_fc)} MPa`],
      ["Coefficient of variation, V<sub>fc,is</sub>", pct(D.V), D.V_min_governs ? `8% minimum governs (measured ${pct(D.V_measured)})` : ""],
      ["Student-t corrected CoV, V<sub>fc,is,corr</sub>", pct(D.V_corr)],
      ["Bias factor, μ<sub>fc,is</sub>", fx(D.bias, 3)],
      ["Number of cores, n", `${D.n} <span class="kv-sub">(k<sub>n</sub> = ${fx(D.k_n, 3)})</span>`],
    ]);

    document.getElementById("dt-detail-1").innerHTML = kvRows([
      { group: "Lognormal parameters" },
      ["Mean of ln f<sub>c</sub>, λ<sub>is</sub>", fx(D.lambda, 4)],
      ["Measured spread, ζ<sub>fc,is</sub>", fx(D.zeta_measured, 4)],
      ["Adopted spread, ζ<sub>fc,is</sub>", fx(D.zeta, 4), D.V_min_governs ? "8% minimum CoV governs" : ""],
      { group: "Coefficient of variation" },
      ["Measured, V<sub>fc,is</sub>", pct(D.V_measured, 2)],
      ["Adopted (8% minimum), V<sub>fc,is</sub>", `<strong>${pct(D.V, 2)}</strong>`],
      ["Student-t corrected, V<sub>fc,is,corr</sub>", pct(D.V_corr, 2), `k<sub>d,n</sub> = ${fx(D.k_dn, 2)} at n = ${D.n}`],
    ]);
    document.getElementById("dt-detail-2").innerHTML = kvRows([
      { group: "Sample size" },
      ["Number of cores, n", D.n],
      ["k<sub>n</sub>", fx(D.k_n, 3)],
      { group: "Result" },
      ["Characteristic, <span class=\"nowrap\">f<sub>ck,is</sub> = f<sub>cm(n),is</sub> · exp(−k<sub>n</sub> V<sub>fc,is</sub>)</span>", `<strong>${fx(D.fck)} MPa</strong>`],
      ["Bias factor, <span class=\"nowrap\">μ<sub>fc,is</sub> = f<sub>cm(n),is</sub> / f<sub>ck,is</sub></span>", fx(D.bias, 3)],
    ]);

    document.getElementById("dt-norm-table").innerHTML = coreNormTable(json.normalisation);
    modalEl.querySelectorAll("details").forEach(d => { d.open = false; });
    showOwningTab(form);
    modal.show();
  }

  /* ---------- plot: core strengths with the mean and characteristic lines ---------- */
  const PURPLE = "#662D91", GREEN_DARK = "#4E7A31", INK = "#1F2328", GRID = "#ECEEF1";
  const CORE_MARK = chartMarker(5, PURPLE, "#fff", 1);
  function drawPlot() {
    const { json } = last;
    const D = json.dt_only;
    const fc = json.normalisation.map(r => r.fc_cyl);
    const n = fc.length;
    const level = (y, label, color, dash) => ({
      label, type: "line", data: [{ x: 0.5, y }, { x: n + 0.5, y }], borderColor: color, backgroundColor: color,
      borderWidth: 2, borderDash: dash, pointRadius: 0, pointStyle: "line", fill: false,
    });
    const canvas = document.getElementById("dt-plot");
    Chart.getChart(canvas)?.destroy();
    new Chart(canvas, {
      type: "scatter",
      data: { datasets: [
        { label: "Your cores", data: fc.map((y, i) => ({ x: i + 1, y })), backgroundColor: PURPLE, borderColor: "#fff",
          borderWidth: 1, pointRadius: 5, pointHoverRadius: 7, legendMarker: CORE_MARK },
        level(D.fcm, `Mean, ${fx(D.fcm)} MPa`, INK, [5, 4]),
        level(D.fck, `Characteristic, ${fx(D.fck)} MPa`, GREEN_DARK, []),
      ] },
      options: {
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { min: 0.5, max: n + 0.5, title: { display: true, text: "Core", font: { size: 12, weight: "500" } },
               grid: { color: GRID },
               afterBuildTicks: ax => { ax.ticks = Array.from({ length: n }, (_, i) => ({ value: i + 1 })); } },
          y: { title: { display: true, text: "Strength, fc,cyl (MPa)", font: { size: 12, weight: "500" } }, grid: { color: GRID } },
        },
        plugins: {
          legend: legendWithMarkers(),
          tooltip: { filter: c => c.datasetIndex === 0,
                     callbacks: { label: c => `Core ${c.parsed.x}: ${fx(c.parsed.y)} MPa` } },
        },
      },
    });
  }
  modalEl.addEventListener("shown.bs.modal", () => { if (last) drawPlot(); });

  /* ---------- footer actions ---------- */
  function summaryText() {
    const D = last.json.dt_only;
    return [`Test region, cores only: n = ${D.n} cores (converted to 150 x 300 mm cylinder)`,
      `fck,is = ${fx(D.fck)} MPa; fcm(n),is = ${fx(D.fcm)} MPa; s = ${fx(D.s_fc)} MPa; ` +
      `V = ${pct(D.V)}${D.V_min_governs ? " (8% minimum)" : ""}; Vcorr = ${pct(D.V_corr)}; bias = ${fx(D.bias, 3)}; k_n = ${fx(D.k_n, 3)}`,
      "Core strengths, fc,cyl (MPa): " + last.json.normalisation.map(r => fx(r.fc_cyl)).join(", ")].join("\n");
  }
  document.getElementById("dt-copy").addEventListener("click", e =>
    copyWithFeedback(e.currentTarget, summaryText(), '<i class="bi bi-clipboard" aria-hidden="true"></i> Copy summary'));

  // keep the table through a refresh
  draft = createDraft("dt-only", { form, tables: { cores } });
  draft.restore();

  document.getElementById("dt-new").addEventListener("click", () => {
    draft.clear();
    modal.hide();
    cores.reset();
    summary.hidden = true;
    serverError.hidden = true;
    document.getElementById("d-cores-errors").hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}
