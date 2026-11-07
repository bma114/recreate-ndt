document.addEventListener("DOMContentLoaded", initIndividual);

/* ==========================================================================
   Individual predictions tab (uses the shared helpers in forms.js)
   ========================================================================== */

// Cross-validated performance, Matthews et al. (2025) Table 5
const INDIV_METRICS = {
  upv:    { name: "Ultrasonic pulse velocity", rmse: 4.24, mae: 2.96, lo: 8.3,  hi: 61.7 },
  rh:     { name: "Rebound hammer",            rmse: 7.53, mae: 5.26, lo: 11.5, hi: 84.3 },
  sonreb: { name: "SonReb (UPV + RH)",         rmse: 3.17, mae: 2.19, lo: 11.1, hi: 54.8 },
};

function initIndividual() {
  const form = document.getElementById("indiv-form");
  if (!form) return;

  const fieldsWrap  = document.getElementById("indiv-fields");
  const upvBlock    = document.getElementById("indiv-upv-block");
  const rhBlock     = document.getElementById("indiv-rh-block");
  const summary     = document.getElementById("indiv-error-summary");
  const serverError = document.getElementById("indiv-server-error");
  const predictBtn  = document.getElementById("predict-btn");
  const modalEl     = document.getElementById("predictModal");
  const modal       = bootstrap.Modal.getOrCreateInstance(modalEl);
  returnFocusOnClose(modalEl, predictBtn);
  const methodRadios = form.querySelectorAll('input[name="ndt"]');
  let lastResultText = "";

  const kit = createFieldKit({ form, scope: fieldsWrap, summary, onEdit: () => { serverError.hidden = true; } });
  const currentNdt = () => form.querySelector('input[name="ndt"]:checked')?.value || "";

  const syncUpvOther = kit.bindOther(document.getElementById("upv_standard"),
    document.getElementById("custom-upv-container"), document.getElementById("custom-upv"), "other_upv");
  const syncRhOther = kit.bindOther(document.getElementById("rh_standard"),
    document.getElementById("custom-rh-container"), document.getElementById("custom-rh"), "other_rh");

  /* ---------- show the blocks for the chosen method ---------- */
  function updateBlocks() {
    const ndt = currentNdt();
    fieldsWrap.hidden = !ndt;
    const showUpv = ndt === "upv" || ndt === "sonreb";
    const showRh  = ndt === "rh"  || ndt === "sonreb";
    upvBlock.hidden = !showUpv;
    rhBlock.hidden  = !showRh;
    // Hidden blocks: disable their controls so they are neither validated nor sent
    upvBlock.querySelectorAll("input, select").forEach(el => { el.disabled = !showUpv; if (!showUpv) kit.clearField(el); });
    rhBlock.querySelectorAll("input, select").forEach(el => { el.disabled = !showRh;  if (!showRh)  kit.clearField(el); });
    syncUpvOther(); syncRhOther();
    document.getElementById("ndt-error").hidden = true;
    form.querySelector(".method-grid").classList.remove("is-invalid");
    if (!summary.hidden) kit.refresh();
  }
  methodRadios.forEach(r => r.addEventListener("change", updateBlocks));

  function allProblems(show) {
    if (!currentNdt()) {
      if (show) {
        const e = document.getElementById("ndt-error");
        e.textContent = "Select a test method."; e.hidden = false;
        form.querySelector(".method-grid").classList.add("is-invalid");
      }
      return [{ id: "method-title", focus: methodRadios[0], msg: "Select a test method." }];
    }
    return kit.fieldProblems(show);
  }
  kit.refresh = () => kit.renderSummary(allProblems(false));

  /* ---------- submit ---------- */
  form.addEventListener("submit", async e => {
    e.preventDefault();
    kit.submitted = true;
    serverError.hidden = true;
    const problems = allProblems(true);
    kit.showSummary(problems);
    if (problems.length) return;

    const ndt = currentNdt();
    const features = {};
    const used = [];
    kit.controls().forEach(el => {
      const v = el.value.trim();
      if (!v || el.id.startsWith("custom-")) return;
      let value = v;
      if (el.id === "upv_standard" && v === "other_upv") value = document.getElementById("custom-upv").value.trim();
      if (el.id === "rh_standard"  && v === "other_rh")  value = document.getElementById("custom-rh").value.trim();
      if (isNum(el)) value = String(parseNum(v));
      features[el.name] = isNum(el) ? parseNum(v) : value;
      const label = el.closest(".field").querySelector("label");
      const shown = el.tagName === "SELECT" && !v.startsWith("other_") ? el.selectedOptions[0].textContent : value;
      const unit = el.closest(".input-group")?.querySelector(".input-group-text")?.textContent || "";
      used.push([labelHtml(label), `${shown}${unit ? " " + unit : ""}`]);
    });
    const notes = kit.notes();

    setButtonBusy(predictBtn, true, "Predict strength", "Predicting…");
    try {
      const json = await postJson("/predict/", { ndt, features });
      showResult(ndt, json.value, used, notes);
    } catch (err) {
      showServerError(serverError, "The prediction could not be made.", err);
    } finally {
      setButtonBusy(predictBtn, false, "Predict strength", "Predicting…");
    }
  });

  /* ---------- result ---------- */
  function showResult(ndt, value, used, notes) {
    const m = INDIV_METRICS[ndt];
    document.getElementById("result-method").textContent = m.name;
    document.getElementById("predict-result").textContent = fmt(value);
    document.getElementById("result-rmse").textContent = `± ${fmt(m.rmse, 2)} MPa`;
    document.getElementById("result-mae").textContent  = `${fmt(m.mae, 2)} MPa`;
    document.getElementById("result-range").textContent = `${fmt(m.lo)}–${fmt(m.hi)} MPa`;

    // gauge: 0 to a round number above both the range and the prediction
    const top = Math.ceil(Math.max(m.hi * 1.15, value * 1.1) / 10) * 10;
    const pct = x => `${Math.min(100, Math.max(0, x / top * 100))}%`;
    const gauge = modalEl.querySelector(".range-gauge");
    gauge.querySelector(".range-band").style.left  = pct(m.lo);
    gauge.querySelector(".range-band").style.width = `calc(${pct(m.hi)} - ${pct(m.lo)})`;
    gauge.querySelector(".range-marker").style.left = pct(value);
    gauge.querySelector(".range-max").textContent = `${top} MPa`;
    const outside = value < m.lo || value > m.hi;
    gauge.classList.toggle("is-outside", outside);

    const warnings = [];
    if (outside) warnings.push(`The prediction lies outside the reliable range for this model (${fmt(m.lo)}–${fmt(m.hi)} MPa), so the model is extrapolating.`);
    warnings.push(...notes);
    const wBox = document.getElementById("result-warnings");
    wBox.querySelector("ul").innerHTML = warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");
    wBox.hidden = !warnings.length;

    modalEl.querySelector(".result-inputs").open = false;
    document.getElementById("result-input-count").textContent = `(${used.length})`;
    document.getElementById("result-input-table").innerHTML =
      used.map(([k, v]) => `<tr><th scope="row">${k}</th><td>${escapeHtml(v)}</td></tr>`).join("");

    lastResultText = `${m.name}: predicted fc,cyl = ${fmt(value)} MPa (150 x 300 mm cylinder). ` +
      `Model RMSE ${fmt(m.rmse, 2)} MPa; reliable range ${fmt(m.lo)}-${fmt(m.hi)} MPa.\n` +
      used.map(([k, v]) => `${k.replace(/<[^>]+>/g, "")}: ${v}`).join("\n");

    showOwningTab(form);
    modal.show();
  }

  document.getElementById("result-copy").addEventListener("click", e =>
    copyWithFeedback(e.currentTarget, lastResultText, '<i class="bi bi-clipboard" aria-hidden="true"></i> Copy result'));

  document.getElementById("result-new").addEventListener("click", () => {
    modal.hide();
    resetForm();
  });

  /* ---------- reset ---------- */
  let clearDraft = () => {};
  function resetForm() {
    clearDraft();
    form.reset();
    kit.submitted = false;
    summary.hidden = true;
    serverError.hidden = true;
    form.querySelectorAll("input, select").forEach(el => kit.clearField(el));
    updateBlocks();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  document.getElementById("indiv-reset").addEventListener("click", resetForm);

  updateBlocks();

  // keep the inputs through a refresh
  const draft = createDraft("individual", { form, radioName: "ndt" });
  draft.restore();
  clearDraft = draft.clear;
}
