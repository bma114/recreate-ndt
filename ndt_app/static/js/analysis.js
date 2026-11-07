/* ==========================================================================
   Test region analysis: combined DT + NDT tab (uses forms.js)
   ========================================================================== */

const METHOD_NAMES = { upv: "Ultrasonic pulse velocity", rh: "Rebound hammer", sonreb: "SonReb (UPV + RH)" };

// 5-95 % range of the training strengths for each model (Matthews et al. 2025, Table 5)
const TRAINING_FC = { upv: [8.3, 61.7], rh: [11.5, 84.3], sonreb: [11.1, 54.8] };

// Table columns (hard limits block the analysis; soft limits give an amber note)
const COLS = {
  vp:  { key: "vp",  label: "Pulse velocity, V<sub>p</sub>", short: "Pulse velocity", unit: "m/s", min: 1000, max: 7000,
         soft: x => rangeNote(x, 2300, 5250, "m/s", "Predictions") },
  rn:  { key: "rn",  label: "Rebound number, RN", short: "Rebound number", unit: "", min: 10, max: 100,
         soft: x => rangeNote(x, 15, 65, "", "Predictions") },
  ...CORE_COLS,
};

document.addEventListener("DOMContentLoaded", initCombined);

function initCombined() {
  const form = document.getElementById("combined-form");
  if (!form) return;

  const fieldsWrap  = document.getElementById("combined-fields");
  const upvBlock    = document.getElementById("c-upv-block");
  const rhBlock     = document.getElementById("c-rh-block");
  const summary     = document.getElementById("combined-error-summary");
  const serverError = document.getElementById("combined-server-error");
  const analyseBtn  = document.getElementById("analyse-btn");
  const modalEl     = document.getElementById("analysisModal");
  const modal       = bootstrap.Modal.getOrCreateInstance(modalEl);
  returnFocusOnClose(modalEl, analyseBtn);
  const methodRadios = form.querySelectorAll('input[name="combined_ndt"]');
  const currentNdt = () => form.querySelector('input[name="combined_ndt"]:checked')?.value || "";
  let last = null;            // last analysis: { json, ndt, notes }

  let draft = null;
  const onEdit = () => { serverError.hidden = true; if (!summary.hidden) kit.refresh(); draft?.queue(); };
  const kit = createFieldKit({ form, scope: fieldsWrap, summary, noteWhat: "The global calibration", onEdit });
  // the tables have their own validation
  fieldsWrap.querySelectorAll(".table-card").forEach(c => c.classList.add("no-kit"));

  const tableOpts = key => ({
    table: document.getElementById(`c-${key}-table`),
    idPrefix: `c-${key}`,
    countEl: document.getElementById(`c-${key}-count`),
    statusEl: document.getElementById(`c-${key}-status`),
    errorsEl: document.getElementById(`c-${key}-errors`),
    notesEl: document.getElementById(`c-${key}-notes`),
    onEdit,
  });
  const calib = createDataTable({ ...tableOpts("calib"), minRows: 8, what: "cores" });
  const comp  = createDataTable({ ...tableOpts("comp"),  minRows: 3, what: "readings" });
  const tables = { calib, comp };
  form.querySelectorAll("[data-add-row]").forEach(b => b.addEventListener("click", () => tables[b.dataset.addRow].addBlankRow()));
  form.querySelectorAll("[data-clear-table]").forEach(b => b.addEventListener("click", () => tables[b.dataset.clearTable].clear()));

  const syncUpvOther = kit.bindOther(document.getElementById("c-upv_standard"),
    document.getElementById("c-custom-upv-container"), document.getElementById("c-custom-upv"), "other_upv");
  const syncRhOther = kit.bindOther(document.getElementById("c-rh_standard"),
    document.getElementById("c-custom-rh-container"), document.getElementById("c-custom-rh"), "other_rh");

  /* ---------- method: show blocks and set table columns ---------- */
  function updateMethod() {
    const ndt = currentNdt();
    fieldsWrap.hidden = !ndt;
    if (!ndt) return;
    const hasUpv = ndt === "upv" || ndt === "sonreb";
    const hasRh  = ndt === "rh"  || ndt === "sonreb";
    upvBlock.hidden = !hasUpv;
    rhBlock.hidden  = !hasRh;
    upvBlock.querySelectorAll("input, select").forEach(el => { el.disabled = !hasUpv; if (!hasUpv) kit.clearField(el); });
    rhBlock.querySelectorAll("input, select").forEach(el => { el.disabled = !hasRh;  if (!hasRh)  kit.clearField(el); });
    syncUpvOther(); syncRhOther();

    const ndtCols = [...(hasUpv ? [COLS.vp] : []), ...(hasRh ? [COLS.rn] : [])];
    calib.setColumns([...ndtCols, COLS.d, COLS.h, COLS.fc]);
    comp.setColumns(ndtCols);

    document.getElementById("c-ndt-error").hidden = true;
    form.querySelector(".method-grid").classList.remove("is-invalid");
    if (!summary.hidden) kit.refresh();
  }
  methodRadios.forEach(r => r.addEventListener("change", updateMethod));

  function allProblems(show) {
    if (!currentNdt()) {
      if (show) {
        const e = document.getElementById("c-ndt-error");
        e.textContent = "Select a test method."; e.hidden = false;
        form.querySelector(".method-grid").classList.add("is-invalid");
      }
      return [{ id: "c-method-title", focus: methodRadios[0], msg: "Select a test method." }];
    }
    const p = kit.fieldProblems(show);
    const c = calib.validate(show).problems.map(x => ({ ...x, msg: x.table ? `Calibration set: ${lc(x.msg)}` : `Calibration set, ${lc(x.msg)}` }));
    const m = comp.validate(show).problems.map(x => ({ ...x, msg: x.table ? `Additional NDT set: ${lc(x.msg)}` : `Additional NDT set, ${lc(x.msg)}` }));
    return [...p, ...c, ...m];
  }
  kit.refresh = () => kit.renderSummary(allProblems(false));
  const lc = s => s.charAt(0).toLowerCase() + s.slice(1);

  /* ---------- submit ---------- */
  form.addEventListener("submit", async e => {
    e.preventDefault();
    kit.submitted = calib.submitted = comp.submitted = true;
    serverError.hidden = true;
    const problems = allProblems(true);
    kit.showSummary(problems);
    if (problems.length) return;

    const ndt = currentNdt();
    const payload = { combined_ndt: ndt };
    kit.controls().forEach(el => {
      const v = el.value.trim();
      if (!v || el.id.includes("custom-")) return;
      let value = isNum(el) ? parseNum(v) : v;
      if (v === "other_upv") value = document.getElementById("c-custom-upv").value.trim();
      if (v === "other_rh")  value = document.getElementById("c-custom-rh").value.trim();
      payload[el.name] = value;
    });
    const n = calib.validate(false).complete, m = comp.validate(false).complete;
    const hasUpv = ndt !== "rh", hasRh = ndt !== "upv";
    if (hasUpv) { payload["velocity_n[]"] = n.map(r => r.vp); payload["velocity_m[]"] = m.map(r => r.vp); }
    if (hasRh)  { payload["rebound_number_n[]"] = n.map(r => r.rn); payload["rebound_number_m[]"] = m.map(r => r.rn); }
    payload["width_diameter_n[]"] = n.map(r => r.d);
    payload["height_n[]"] = n.map(r => r.h);
    payload["fc_n[]"] = n.map(r => r.fc);

    const notes = [...kit.notes(),
      // core diameters outside 44-160 mm are reported by the server, so skip the duplicate note
      ...calib.validate(false).notes.filter(x => !/Core diameter/.test(x)).map(x => `Calibration set, ${lc(x)}`),
      ...comp.validate(false).notes.map(x => `Additional NDT set, ${lc(x)}`)];

    setButtonBusy(analyseBtn, true, "Run analysis", "Analysing…");
    try {
      const json = await postJson("/analyse/", payload);
      last = { json, ndt, notes, n: n.length, m: m.length, input: { n, m } };
      showResults();
    } catch (err) {
      showServerError(serverError, "The analysis could not be run.", err);
    } finally {
      setButtonBusy(analyseBtn, false, "Run analysis", "Analysing…");
    }
  });

  /* ---------- results ---------- */
  function kvRows(rows) {
    return rows.map(r => r.group
      ? `<tr class="kv-group"><th colspan="2">${r.group}</th></tr>`
      : `<tr><th scope="row">${r[0]}</th><td>${r[1]}${r[2] ? `<span class="kv-note${/governs/.test(r[2]) ? " kv-warn" : ""}">${r[2]}</span>` : ""}</td></tr>`).join("");
  }

  function summaryRows(R, extra = []) {
    return kvRows([
      ["Mean in-situ strength, f<sub>cm,is</sub>", `${fx(R.fcm)} MPa`],
      ["Coefficient of variation, V<sub>fc,is</sub>", pct(R.V), R.dt_governs ? "cores-only value governs" : ""],
      ["Student-t corrected CoV, V<sub>fc,is,corr</sub>", pct(R.V_corr)],
      ["Bias factor, μ<sub>fc,is</sub>", fx(R.bias, 3)],
      ["Effective sample size, n<sub>eff</sub>", `${R.n_eff} <span class="kv-sub">(k<sub>n,eff</sub> = ${fx(R.k_n_eff, 3)})</span>`],
      ...extra,
    ]);
  }

  const varName = k => k.endsWith("Vp") ? "V<sub>p</sub>" : "RN";
  function equationHtml(c) {
    const terms = Object.entries(c).filter(([k]) => k !== "a")
      .map(([k, v]) => `${v >= 0 ? "+" : "−"} ${fx(Math.abs(v), 4)}&nbsp;ln&nbsp;${varName(k)}`).join(" ");
    return `ln f<sub>c,cyl</sub> = ${c.a < 0 ? "−" : ""}${fx(Math.abs(c.a), 4)} ${terms}`;
  }
  // the same relationship back-transformed: f = A · Vp^b · RN^c, with A = exp(a)
  function sci(x) {
    if (x >= 0.01 && x < 1e5) return fx(x, x < 1 ? 4 : x < 100 ? 3 : 1);
    const e = Math.floor(Math.log10(x));
    return `${fx(x / 10 ** e, 3)}&nbsp;×&nbsp;10<sup>${e < 0 ? "−" : ""}${Math.abs(e)}</sup>`;
  }
  function powerHtml(c) {
    const terms = Object.entries(c).filter(([k]) => k !== "a")
      .map(([k, v]) => `${varName(k)}<sup>${v < 0 ? "−" : ""}${fx(Math.abs(v), 3)}</sup>`).join("&nbsp;·&nbsp;");
    return `f<sub>c,cyl</sub> = ${sci(Math.exp(c.a))}&nbsp;·&nbsp;${terms}`;
  }

  function detailRows(R, isGlobal) {
    const rows = [];
    rows.push({ group: "Lognormal parameters" });
    rows.push(["Mean of ln f<sub>c</sub>, λ<sub>(m),is</sub>", fx(R.lambda, 4)]);
    rows.push(["Spread of the additional set, ζ<sub>fc,is,test</sub>", fx(R.zeta_test, 4)]);
    rows.push(["Model uncertainty, ζ<sub>θ,test</sub>", fx(R.zeta_theta, 4),
      R.floor_governs ? `2 MPa minimum governs (calculated ${fx(R.zeta_theta_raw, 4)})` : ""]);
    rows.push(["Combined, ζ<sub>fc,is</sub>", fx(R.zeta_cal, 4)]);
    rows.push({ group: "Coefficient of variation" });
    rows.push(["From the calibration, V<sub>fc,is</sub>", pct(R.V_cal, 2)]);
    rows.push(["Cores only, V<sub>fc,is,DT</sub>", pct(R.V_dt, 2)]);
    rows.push(["Adopted (the larger)", `<strong>${pct(R.V, 2)}</strong>`]);
    rows.push(["Student-t corrected, V<sub>fc,is,corr</sub>", pct(R.V_corr, 2), `k<sub>d,n</sub> = ${fx(R.k_dn, 2)} at n = ${R.n}`]);
    rows.push({ group: "Sample size" });
    rows.push(["Cores n, additional readings m", `${R.n}, ${R.m}`]);
    rows.push(["Effective sample size, n<sub>eff</sub>", R.n_eff]);
    rows.push(["k<sub>n,eff</sub>", fx(R.k_n_eff, 3)]);
    rows.push({ group: "Result" });
    rows.push(["Characteristic, <span class=\"nowrap\">f<sub>ck,is</sub> = exp(λ − k<sub>n,eff</sub>ζ)</span>", `<strong>${fx(R.fck)} MPa</strong>`]);
    rows.push(["Mean, <span class=\"nowrap\">f<sub>cm,is</sub> = exp(λ + ζ²/2)</span>", `${fx(R.fcm)} MPa`]);
    rows.push(["Bias factor, μ<sub>fc,is</sub>", fx(R.bias, 3)]);
    if (isGlobal) {
      rows.push({ group: "Shift to your cores" });
      rows.push(["Mean log-residual, Δ", fx(R.shift, 4), `model predictions × ${fx(Math.exp(R.shift), 3)}`]);
    }
    return kvRows(rows);
  }

  function locationTable(R, isGlobal) {
    const { ndt, input } = last;
    const hasUpv = ndt !== "rh", hasRh = ndt !== "upv";
    const fcCyl = last.json.normalisation.map(r => r.fc_cyl);
    const head = `<thead><tr><th class="col-idx">#</th>${hasUpv ? "<th>V<sub>p</sub> (m/s)</th>" : ""}${hasRh ? "<th>RN</th>" : ""}
      <th>Core f<sub>c,cyl</sub> (MPa)</th>${isGlobal ? "<th>ML model, before shift (MPa)</th>" : ""}
      <th class="col-pred">Calibrated prediction (MPa)</th><th>Prediction error, MPa (%)</th></tr></thead>`;
    const ndtCells = r => `${hasUpv ? `<td>${fmtPlain(r.vp)}</td>` : ""}${hasRh ? `<td>${fmtPlain(r.rn)}</td>` : ""}`;
    const err = (p, f) => { const d = p - f; return `${d > 0 ? "+" : d < 0 ? "−" : ""}${fx(Math.abs(d))} <span class="kv-sub">(${d > 0 ? "+" : d < 0 ? "−" : ""}${fx(Math.abs(d) / f * 100, 0)}%)</span>`; };
    const nRows = input.n.map((r, i) => `<tr><td class="col-idx">${i + 1}</td>${ndtCells(r)}<td>${fx(fcCyl[i])}</td>
      ${isGlobal ? `<td>${fx(R.pred_n_unshifted[i])}</td>` : ""}<td class="col-pred">${fx(R.pred_n[i])}</td><td>${err(R.pred_n[i], fcCyl[i])}</td></tr>`).join("");
    const mRows = input.m.map((r, i) => `<tr><td class="col-idx">${i + 1}</td>${ndtCells(r)}<td>–</td>
      ${isGlobal ? `<td>${fx(R.pred_m_unshifted[i])}</td>` : ""}<td class="col-pred">${fx(R.pred_m[i])}</td><td>–</td></tr>`).join("");
    const span = 4 + (hasUpv ? 1 : 0) + (hasRh ? 1 : 0) + (isGlobal ? 1 : 0);
    return `${head}<tbody><tr class="kv-group"><th colspan="${span}">Calibration set (cores)</th></tr>${nRows}
      <tr class="kv-group"><th colspan="${span}">Additional NDT set</th></tr>${mRows}</tbody>`;
  }

  function showResults() {
    const { json, ndt, notes } = last;
    const L = json.local, G = json.global, D = json.dt_only;

    document.getElementById("an-method").textContent = `Test region · ${METHOD_NAMES[ndt]}`;
    document.getElementById("an-sub").textContent =
      `${last.n} cores with paired NDT, ${last.m} additional NDT readings`;

    // warnings: server notes first, then the amber notes from the form
    const warnings = [...(json.warnings || []), ...notes];
    const wBox = document.getElementById("an-warnings");
    wBox.querySelector("ul").innerHTML = warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");
    wBox.hidden = !warnings.length;

    document.getElementById("an-local-fck").textContent = fx(L.fck);
    document.getElementById("an-global-fck").textContent = fx(G.fck);
    document.getElementById("an-local-summary").innerHTML =
      summaryRows(L, [["Regression fit, R<sup>2</sup> (log space)", fx(L.R2_log, 3)]]);
    document.getElementById("an-global-summary").innerHTML =
      summaryRows(G, [["Shift applied to the model", `× ${fx(Math.exp(G.shift), 3)}`]]);

    // Cores-only reference: shown only when its CoV governs one of the calibrations
    const govern = [L.dt_governs && "local", G.dt_governs && "global"].filter(Boolean);
    document.getElementById("an-dt-ref").hidden = !govern.length;
    document.getElementById("an-dt-why").textContent = govern.length
      ? `The cores alone show more variability than the ${govern.join(" and ")} calibration${govern.length > 1 ? "s" : ""}, so their coefficient of variation is adopted there: a calibration never reports less variability than the cores themselves.`
      : "";
    document.getElementById("an-dt-summary").innerHTML = [
      ["f<sub>ck,is</sub>", `${fx(D.fck)} MPa`],
      ["f<sub>cm,is</sub>", `${fx(D.fcm)} MPa`],
      ["V<sub>fc,is</sub>", pct(D.V) + (D.V_min_governs ? " <small>(8% minimum)</small>" : "")],
      ["V<sub>fc,is,corr</sub>", pct(D.V_corr)],
      ["μ<sub>fc,is</sub>", fx(D.bias, 3)],
      ["k<sub>n</sub>", fx(D.k_n, 3)],
    ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");

    document.getElementById("an-norm-table").innerHTML = coreNormTable(json.normalisation);

    const eq = document.getElementById("an-local-equation");
    eq.innerHTML = `<span class="eq-label">Fitted relationship</span><span class="eq">${powerHtml(L.coefficients)}</span>
      <span class="eq-alt">Fitted as ${equationHtml(L.coefficients)}</span>
      <span class="eq-r2">R<sup>2</sup> = ${fx(L.R2_log, 3)} (log space)</span>`;
    eq.hidden = false;
    document.getElementById("an-local-detail").innerHTML = detailRows(L, false);
    document.getElementById("an-global-detail").innerHTML = detailRows(G, true);
    document.getElementById("an-local-locations").innerHTML = locationTable(L, false);
    document.getElementById("an-global-locations").innerHTML = locationTable(G, true);
    modalEl.querySelectorAll("details").forEach(d => { d.open = false; });

    bootstrap.Tab.getOrCreateInstance(document.getElementById("summary-tab")).show();
    showOwningTab(form);
    modal.show();
  }

  // Summary cards link to their detail tabs
  modalEl.querySelectorAll("[data-goto-tab]").forEach(b => b.addEventListener("click", () =>
    bootstrap.Tab.getOrCreateInstance(document.getElementById(b.dataset.gotoTab)).show()));

  /* ---------- charts (drawn when their tab opens) ---------- */
  const PURPLE = "#662D91", GREEN = "#72A64B", GREEN_DARK = "#4E7A31", INK = "#1F2328", GRID = "#ECEEF1";
  if (window.Chart) {
    Chart.defaults.font.family = "'Poppins', system-ui, sans-serif";
    Chart.defaults.color = "#4A5058";
  }
  const DB_FILL = "rgba(102,45,145,.38)";
  const MARK = {
    local: chartMarker(5, PURPLE, "#fff", 1),
    global: chartMarker(5.5, GREEN, INK, 1.2),
    db: chartMarker(2.75, DB_FILL, null, 0),
  };
  const axis = text => ({ title: { display: true, text, font: { size: 12, weight: "500" } }, grid: { color: GRID } });

  function draw(id, datasets, xLabel, yLabel) {
    const canvas = document.getElementById(id);
    Chart.getChart(canvas)?.destroy();
    new Chart(canvas, {
      type: "scatter",
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        animation: false,
        scales: { x: axis(xLabel), y: { ...axis(yLabel), beginAtZero: false } },
        plugins: {
          legend: legendWithMarkers(),
          tooltip: { callbacks: { label: c => `${c.dataset.label}: (${fmtPlain(c.parsed.x)}, ${fmtPlain(c.parsed.y)})` } },
        },
      },
    });
  }

  const Y_LABEL = "Strength, fc,cyl (MPa)";
  const X = { vp: "Pulse velocity, Vp (m/s)", rn: "Rebound number, RN" };

  function plotSet(ndt) {
    return ndt === "sonreb" ? [["vp", "Vp", "Xn_Vp", "curve_Vp", "X_global_Vp"], ["rn", "RN", "Xn_RN", "curve_RN", "X_global_RN"]]
         : ndt === "upv"    ? [["vp", "Vp", "Xn_Vp", "curve_Vp", "X_global"]]
                            : [["rn", "RN", "Xn_RN", "curve_RN", "X_global"]];
  }

  function drawLocal() {
    const { json, ndt } = last;
    const lp = json.local_plots;
    const sets = plotSet(ndt);
    [1, 2].forEach(i => { document.getElementById(`local-plot-${i}-box`).hidden = i > sets.length; });
    sets.forEach(([k, , xn, cv], i) => {
      const pts = lp[xn].map((x, j) => ({ x, y: lp.Y[j] }));
      const curve = lp[cv] ? lp[cv].x.map((x, j) => ({ x, y: lp[cv].y[j] })) : [];
      draw(`local-plot-${i + 1}`, [
        { label: "Your cores", data: pts, backgroundColor: PURPLE, borderColor: "#fff", borderWidth: 1, pointRadius: 5, pointHoverRadius: 7, legendMarker: MARK.local },
        { label: "Log-log fit", data: curve, type: "line", borderColor: GREEN_DARK, backgroundColor: GREEN_DARK, borderWidth: 2.5, pointRadius: 0, pointStyle: "line", fill: false },
      ], X[k], Y_LABEL);
    });
    document.getElementById("local-plot-caption").textContent = ndt === "sonreb"
      ? "Each curve holds the other reading at the geometric mean of your cores."
      : "The fit is valid only within the range of your readings.";
  }

  function drawGlobal() {
    const { json, ndt } = last;
    const gp = json.global_plots, lp = json.local_plots;
    const sets = plotSet(ndt);
    [1, 2].forEach(i => { document.getElementById(`global-plot-${i}-box`).hidden = i > sets.length; });
    sets.forEach(([k, , xn, , xg], i) => {
      const db = gp[xg].map((x, j) => ({ x, y: gp.Y_global[j] }));
      const you = lp[xn].map((x, j) => ({ x, y: lp.Y[j] }));
      draw(`global-plot-${i + 1}`, [
        { label: "Your cores", data: you, backgroundColor: GREEN, borderColor: INK, borderWidth: 1.2, pointRadius: 5.5, pointHoverRadius: 7, order: 0, legendMarker: MARK.global },
        { label: `Training database (${db.length.toLocaleString("en-GB")})`, data: db, backgroundColor: DB_FILL,
          borderWidth: 0, pointRadius: 2.75, pointHoverRadius: 4, order: 1, legendMarker: MARK.db },
      ], X[k], Y_LABEL);
    });
    document.getElementById("global-plot-caption").textContent =
      "Your cores against the open database used to train the model.";
  }

  document.getElementById("analysisTabs").addEventListener("shown.bs.tab", e => {
    if (!last) return;
    if (e.target.id === "local-tab") drawLocal();
    if (e.target.id === "global-tab") drawGlobal();
  });

  /* ---------- footer actions ---------- */
  function summaryText() {
    const { json, ndt } = last;
    const L = json.local, G = json.global, D = json.dt_only;
    const line = (name, R) => `${name}: fck,is = ${fx(R.fck)} MPa; fcm,is = ${fx(R.fcm)} MPa; V = ${pct(R.V)}; ` +
      `Vcorr = ${pct(R.V_corr)}; bias = ${fx(R.bias, 3)}; n_eff = ${R.n_eff}; k_n,eff = ${fx(R.k_n_eff, 3)}`;
    return [`Test region, ${METHOD_NAMES[ndt]}: n = ${last.n} cores, m = ${last.m} additional readings`,
      line("Local calibration", L), line("Global calibration", G),
      `Cores only: fck,is = ${fx(D.fck)} MPa; V = ${pct(D.V)}; Vcorr = ${pct(D.V_corr)}; bias = ${fx(D.bias, 3)}`].join("\n");
  }
  document.getElementById("an-copy").addEventListener("click", e =>
    copyWithFeedback(e.currentTarget, summaryText(), '<i class="bi bi-clipboard" aria-hidden="true"></i> Copy summary'));

  function resetForm() {
    draft?.clear();
    form.reset();
    kit.submitted = false;
    summary.hidden = true;
    serverError.hidden = true;
    form.querySelectorAll("input, select").forEach(el => { if (!el.classList.contains("cell")) kit.clearField(el); });
    calib.columns = []; comp.columns = [];
    fieldsWrap.hidden = true;
    calib.reset(); comp.reset();
    ["calib", "comp"].forEach(k => { document.getElementById(`c-${k}-errors`).hidden = true; });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  document.getElementById("an-new").addEventListener("click", () => { modal.hide(); resetForm(); });
  document.getElementById("combined-reset").addEventListener("click", () => {
    if (window.confirm("Clear the whole form, including both tables?")) resetForm();
  });

  updateMethod();

  // keep the inputs and both tables through a refresh
  draft = createDraft("combined", { form, tables: { calib, comp }, radioName: "combined_ndt" });
  draft.restore();
}
