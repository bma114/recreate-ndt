document.addEventListener("DOMContentLoaded", function() {    
  const combinedForm = document.getElementById('combined-form');
  const analyseBtn   = document.getElementById('analyse-btn');
  const ndtSelect      = document.getElementById("combined_ndt");
  const featureCard    = document.getElementById("combined-features-card");
  const upvBlock       = document.getElementById("combined-upv-block");
  const rhBlock        = document.getElementById("combined-rh-block");
  const calibHeaders   = document.getElementById("calib-headers");
  const calibBody      = document.getElementById("calib-rows");
  const addCalibBtn    = document.getElementById("add-calib-row");
  const calibContainer = document.getElementById("calibration-set-container");
  const compContainer  = document.getElementById("complementary-container");
  const compHeaders    = document.getElementById("comp-headers");
  const compBody       = document.getElementById("comp-rows");
  const addCompBtn     = document.getElementById("add-comp-row");

  // UPV-specific fields for custom standards
  const upvSelect          = document.getElementById("upv_standard");
  const customUpvContainer = document.getElementById("comb-custom-upv-container");
  const customUpvInput     = document.getElementById("comb-custom-upv");

  // RH-specific fields for custom standards
  const rhSelect           = document.getElementById("rh_standard");
  const customRhContainer  = document.getElementById("comb-custom-rh-container");
  const customRhInput      = document.getElementById("comb-custom-rh");

  async function runAnalysis() {
    analyseBtn.disabled = true;
    analyseBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Please wait...';

    /* analyseCombined() must return a Promise (fetch/await etc.)   */
    try {
      await analyseCombined();        
    } finally {
      analyseBtn.disabled = false;
      analyseBtn.innerText = 'ANALYSE';
    }
  }

  combinedForm.addEventListener('submit', e => {
    e.preventDefault();                     // stop page navigation
    if (validateCombined()) runAnalysis();  // spinner + fetch
  });
 
  // Function to disable fields when hidden
  function setInputsDisabled(container, disabled) {
    const ndt = ndtSelect.value;
    const showUPV = (ndt === 'upv' || ndt === 'sonreb');
    const showRH = (ndt === 'rh' || ndt === 'sonreb');

    // disable (and un-require) everything inside upvBlock
    upvBlock.querySelectorAll('input,select,textarea').forEach(el => {
      el.disabled = !showUPV;
      if (!showUPV) {
        if (el.hasAttribute('required')) el.dataset.wasReq = '1';
        el.removeAttribute('required');
      } else if (!el.hasAttribute('data-optional')) {
        if (el.dataset.wasReq || el.hasAttribute('required')) el.setAttribute('required','');
      }
    });
    if (!showUPV) {
      customUpvContainer.style.display = 'none';
      customUpvInput.removeAttribute('required');
      customUpvInput.value = '';
    }
    
    // disable (and un‐require) everything inside rhBlock
    rhBlock.querySelectorAll('input,select,textarea').forEach(el => {
      el.disabled = !showRH;
      if (!showRH) el.removeAttribute('required');
      else         el.setAttribute('required','');
    });
    if (!showRH) {
      customRhContainer.style.display = 'none';
      customRhInput.removeAttribute('required');
      customRhInput.value = '';
    }
  } 

  setInputsDisabled();
  ndtSelect.addEventListener('change', setInputsDisabled);

  // Function to handle 'Other' option for any select field
  function handleCustomInput(selectEl, customBox, customInput, otherValue) {
    selectEl.addEventListener('change', () => {
      const useOther = selectEl.value === otherValue;
      customBox.style.display = useOther ? 'block' : 'none';

      if (useOther) {
      customInput.setAttribute('required', 'true');
      } else {
      customInput.removeAttribute('required');
      customInput.value = '';
      // clear any prior red message
      const err = customBox.querySelector('.error-message');
      if (err) { err.innerText = ''; err.style.display = 'none'; }
      }
    });
  }

  // Attach event listeners for both UPV and RH fields
  handleCustomInput(upvSelect, customUpvContainer, customUpvInput, "other_upv");
  handleCustomInput(rhSelect, customRhContainer, customRhInput, "other_rh");


  // helper to build headers & rows
  function refreshTables() {
    setInputsDisabled();
    const ndt    = ndtSelect.value;
    const analyseBtn = document.getElementById("analyse-btn");
    const hasUPV = ndt === "upv"    || ndt === "sonreb";
    const hasRH  = ndt === "rh"     || ndt === "sonreb";

    // Clear out old rows to avoid accumulation
    calibBody.innerHTML = "";
    compBody.innerHTML  = "";

    // show feature card & tables
    featureCard.style.display       = "block";
    calibContainer.style.display    = "";
    compContainer.style.display     = "";

    // show/hide feature blocks
    upvBlock.style.display = hasUPV ? "" : "none";
    rhBlock.style.display  = hasRH  ? "" : "none";

    // Now toggle the Analyse button
    if (ndt) {
      analyseBtn.style.display = "inline-block";
      analyseBtn.disabled = false;
    } else {
      analyseBtn.style.display = "none";
      analyseBtn.disabled = true;
    }

    // build a header row based on cols array
    function buildHead(el, cols) {
      el.innerHTML = cols.map(c => {
        if (typeof c === 'string') return `<th>${c}</th>`;
        const cls = c.class ? ` class="${c.class}"` : '';
        return `<th${cls}>${c.text}</th>`;
      }).join('');
    }

    // 1) Calibration set: Vp?, RN?, Core Dia, Core Ht, Raw fₖ
    buildHead(calibHeaders, [
      { class:'w-idx', text:'' },
      ...(hasUPV ? [{ text:'V<sub>p</sub> (m/s)' }] : []),
      ...(hasRH  ? [{ text:'Rebound Number, RN'   }] : []),
      { text:'Core Diameter (mm)' },
      { text:'Core Height (mm)'   },
      { text:'Raw f<sub>c,core</sub> (MPa)' },
      { class:'w-del', text:'' }          // delete column
    ]);

    // seed 8 rows on first change
    if (!calibBody.children.length) for (let i=0; i<8; i++) addCalibRow();

    // 2) Complementary set: same columns as calib
    buildHead(compHeaders, [
      { class:'w-idx', text:'' },
      ...(hasUPV ? [{ class:'w-data', text:'V<sub>p</sub> (m/s)' }] : []),
      ...(hasRH  ? [{ class:'w-data', text:'Rebound Number, RN'   }] : []),
      { class:'w-del', text:'' }
    ]);

    // seed 3 rows on first change
    if (!compBody.children.length) for (let i=0; i<3; i++) addCompRow();
  }

  // Build input cells for tables
  function buildCalibCells(hasUPV, hasRH) {
    let html = '';
    if (hasUPV) html += `<td><input type="number" class="form-control"
                                    name="velocity_n[]"          min="1"></td>`;
    if (hasRH)  html += `<td><input type="number" class="form-control"
                                    name="rebound_number_n[]"    min="0"></td>`;

    html += `
      <td><input type="number" class="form-control" name="width_diameter_n[]" min="1"></td>
      <td><input type="number" class="form-control" name="height_n[]"         min="1"></td>
      <td><input type="number" class="form-control" name="fc_n[]"             min="0"></td>
    `;
    return html;
  }

  /* keep the <th> numbers sequential */
  function renumberRows(tbody) {
    Array.from(tbody.rows).forEach((tr,i) => {
      const numCell = tr.querySelector('.w-idx');
      if (numCell) numCell.innerText = i + 1;
    });
  }

  const MIN_CALIB_ROWS = 8; 
  const MIN_COMP_ROWS = 3;

  // add calibration row
  function addCalibRow() {
    const ndt     = ndtSelect.value;
    const hasUPV  = ndt === 'upv'  || ndt === 'sonreb';
    const hasRH   = ndt === 'rh'   || ndt === 'sonreb';
    const idx     = calibBody.children.length + 1;

    const tr = document.createElement('tr');
    let html  = '';

    /* row number  */
    html += `<th class="w-idx pe-1">${idx}</th>`;
    html += buildCalibCells(hasUPV, hasRH);

    /* delete button cell (only if beyond the mandatory 8 rows) */
    if (idx > MIN_CALIB_ROWS) {
      html += `
        <td class="w-del">
          <button type="button" class="delete-row btn p-0 text-dark">
            <i class="bi bi-dash-circle-fill"></i>
          </button>
        </td>`;
    } else {
      html += '<td class="w-del"></td>';
    }

    tr.innerHTML = html;
    calibBody.appendChild(tr);
  }

  /* click anywhere inside the <tbody id="calib-rows"> */
  calibBody.addEventListener('click', e => {
    if (!e.target.closest('.delete-row')) return;   // not our minus icon
    e.target.closest('tr').remove();
    renumberRows(calibBody);
    highlightEmptyRows(calibBody, MIN_CALIB_ROWS);  // keep blank-row logic
  });

  // add complementary row
  function addCompRow() {
    const ndt = ndtSelect.value;
    const hasUPV = ndt==="upv"||ndt==="sonreb";
    const hasRH  = ndt==="rh" ||ndt==="sonreb";
    const idx     = compBody.children.length + 1;
    const tr = document.createElement('tr');
    let html  = '';

    html += `<th class="w-idx pe-1">${idx}</th>`;
    if (hasUPV) html += `<td><input type="number" class="form-control" name="velocity_m[]" min="1"></td>`;
    if (hasRH)  html += `<td><input type="number" class="form-control" name="rebound_number_m[]" min="0"></td>`;
    if (idx > MIN_COMP_ROWS) {
      html += `<td class="w-del"><button type="button" class="delete-row btn p-0 text-dark">
            <i class="bi bi-dash-circle-fill"></i></button></td>`;
    } else {
      html += '<td class="w-del"></td>';
    }

    tr.innerHTML = html;
    compBody.appendChild(tr);
  }

  function attachRowDelete(tbody, minRows) {
    tbody.addEventListener('click', e => {
      if (!e.target.closest('.delete-row')) return;
      e.target.closest('tr').remove();
      renumberRows(tbody);
      highlightEmptyRows(tbody, minRows);
    });
  }

  attachRowDelete(calibBody, MIN_CALIB_ROWS);
  attachRowDelete(compBody , MIN_COMP_ROWS);

  // Function highlighting empty incomplete table rows 
  function highlightEmptyRows(tbody, minRows, showMsg = false) {
    let anyBlank = false;

    Array.from(tbody.querySelectorAll('tr')).forEach((tr, idx) => {
      const inputs = tr.querySelectorAll('input');
      const blank  = [...inputs].every(inp => inp.value.trim() === '');

      const needsWarn = blank && idx >= minRows;
      // tr.classList.toggle('table-danger', needsWarn);
      anyBlank ||= needsWarn;
      tr.classList.remove('table-danger');
      tr.querySelector('.badge-row-error')?.remove();

      /* --- badge in delete column instead of huge <td> ----------------- */
      if (needsWarn && showMsg) {
      /* add red background */
      tr.classList.add('table-danger');

      /* add tiny badge in delete column */
      const delCell = tr.querySelector('.w-del') || tr.lastElementChild;
      const badge = document.createElement('span');
      badge.className = 'badge-row-error';
      badge.setAttribute('data-bs-toggle', 'tooltip');
      badge.setAttribute('title', 'Row is empty – fill or delete');
      delCell.appendChild(badge);
      bootstrap.Tooltip.getOrCreateInstance(badge);
    }
  });

  return anyBlank;
}

  addCalibBtn.addEventListener('click', () => {
    addCalibRow();
    highlightEmptyRows(calibBody, 8, false);
  });

  addCompBtn.addEventListener('click', () => {
    addCompRow();
    highlightEmptyRows(compBody, 3, false);
  });

  function minRowsFor(tbody) {
    return tbody.id === 'calib-rows' ? 8 : 3;
  }

  // also run once after every manual edit:
  document.querySelectorAll('#calib-rows,#comp-rows').forEach(tbody => 
    tbody.addEventListener('input', () => 
      highlightEmptyRows(tbody, minRowsFor(tbody), false)));


  // Function to enable Excel copy and paste
  function installExcelPaste(bodySelector, addRow) {
    const tbody = document.querySelector(bodySelector);
    tbody.addEventListener("paste", e => {
      e.preventDefault();
      const text = e.clipboardData.getData("text/plain").trim();
      const rows = text.split(/\r?\n/).map(r => r.split("\t"));

      // Which input got the paste?
      const input     = e.target.closest("input");
      if (!input) return;
      const startCell = input.closest("td");
      const startRow  = startCell.parentElement;
      const startIdx  = Array.from(tbody.children).indexOf(startRow);
      const startCol  = Array.from(startRow.cells).indexOf(startCell);

      rows.forEach((rowData, r) => {
        const targetIdx = startIdx + r;
        // if that row doesn't exist yet, add one
        while (tbody.children.length <= targetIdx) {
          addRow();
        }
        const rowElem = tbody.children[targetIdx];
        rowData.forEach((val, c) => {
          const cell = rowElem.cells[startCol + c];
          if (cell) {
            const inp = cell.querySelector("input");
            if (inp) inp.value = val;
          }
        });
      });
    });
  }

    installExcelPaste("#calib-rows", () => addCalibRow());
    installExcelPaste("#comp-rows",  () => addCompRow());

    //  Clear calibration rows
    document.getElementById("clear-calib").addEventListener("click", () => {
      // Keep 8 empty rows
      const rows = document.querySelectorAll("#calib-rows tr");
      rows.forEach((tr, i) => {
        if (i < 8) {
          tr.querySelectorAll("input").forEach(inp => inp.value = "");
        } else {
          tr.remove();
        }
      });
    });

    // Clear complementary rows
    document.getElementById("clear-comp").addEventListener("click", () => {
      // Keep 3 empty rows
      const rows = document.querySelectorAll("#comp-rows tr");
      rows.forEach((tr, i) => {
        if (i < 3) {
          tr.querySelectorAll("input").forEach(inp => inp.value = "");
        } else {
          tr.remove();
        }
      });
    });

    // when NDT changes
    // ndtSelect.addEventListener("change", refreshTables);
    ndtSelect.addEventListener('change', () => {
      const chosen = ndtSelect.value;
      combinedForm.reset();
      ndtSelect.value = chosen;

      setInputsDisabled();
      refreshTables();
    });

    // final submit stub
    window.analyseCombined = async function() {
      const form = document.getElementById("combined-form");
      const data = new FormData(form);

      // Convert FormData → JSON-friendly object
      let payload = {};
      for (let [k,v] of data.entries()) {
        if (!payload[k]) payload[k] = [];
        payload[k].push(v);
      }

      return fetch("/analyse/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
            
      .then(json => {
        window._lastAnalysisJSON = json;
        const selectedNDT = document.getElementById("combined_ndt").value;
        const summLocal  = document.getElementById("summary-local");
        const summGlobal = document.getElementById("summary-global");
        const localDetails = json.local_details;
        const fullLocal    = document.getElementById("local-full-summary");
        const globalDetails = json.global_details;
        const fullGlobal    = document.getElementById("global-full-summary");
      
        // Build output lists
        // Local Summary
        const localSummItems = [
        { label: 'Mean Normalised Strength (n-set): f<sub>cm,n,is</sub> = ', key: 'f<sub>cm,n,is</sub>', unit: 'MPa', precision: 2 },
        { label: 'Mean Prediction (m-set): f<sub>cm,m,is</sub> = ',          key: 'f<sub>cm,m,is</sub>', unit: 'MPa', precision: 2 },
        { label: 'Overall: s<sub>fc,is</sub> = ',                            key: 's<sub>fc,is</sub>', unit: 'MPa', precision: 3 },
        { label: 'CoV: V<sub>fc</sub> = ',                                   key: 'V<sub>fc</sub>', unit: '', precision: 3 },
        { label: 'Characteristic Strength: f<sub>ck,is</sub> = ',            key: 'f<sub>ck,is</sub>', unit: 'MPa', precision: 2 },
        ];
        summLocal.innerHTML = '';
        localSummItems.forEach(item => {
          const raw = json.local_summary[item.key];
          const val = parseFloat(raw).toFixed(item.precision);
          const li  = document.createElement('li');
          li.innerHTML = `${item.label} ${val}${item.unit? ' '+item.unit : ''}`;
          summLocal.appendChild(li);
        });

        // Global Summary
        const GlobalSummItems = [
        { label: 'Mean Normalised Strength (n-set): f<sub>cm,n,is</sub> = ', key: 'f<sub>cm,n,is</sub>', unit: 'MPa', precision: 2 },
        { label: 'Mean Prediction (m-set): f<sub>cm,m,is</sub> = ',          key: 'f<sub>cm,m,is</sub>', unit: 'MPa', precision: 2 },
        { label: 'Overall: s<sub>fc,is</sub> = ',                            key: 's<sub>fc,is</sub>', unit: 'MPa', precision: 3 },
        { label: 'CoV: V<sub>fc</sub> = ',                                   key: 'V<sub>fc</sub>', unit: '', precision: 3 },
        { label: 'Characteristic Strength: f<sub>ck,is</sub> = ',            key: 'f<sub>ck,is</sub>', unit: 'MPa', precision: 2 },
        ];
        summGlobal.innerHTML = '';
        GlobalSummItems.forEach(item => {
          const raw = json.global_summary[item.key];
          const val = parseFloat(raw).toFixed(item.precision);
          const li  = document.createElement('li');
          li.innerHTML = `${item.label} ${val}${item.unit? ' '+item.unit : ''}`;
          summGlobal.appendChild(li);
        });

        // Local Tab (Full Details)
        fullLocal.innerHTML = '';
        // Section: Predictions
        const predsSection_lr = `
          <li><strong>Calibration Set Predictions:</strong></li>
          <li>f<sub>c,n,is,pred</sub> (MPa) = [
              ${(json.local_details.n_pred  || [])
                  .map(v => v.toFixed(2))        
                  .join(", ") }]</li>
          <li><strong>Complementary Set Predictions:</strong></li>
          <li>f<sub>c,m,is,pred</sub> (MPa) = [
              ${(json.local_details.m_pred  || [])
                  .map(v => v.toFixed(2))
                  .join(", ") }]</li>`;
        fullLocal.insertAdjacentHTML("beforeend", predsSection_lr);
        // Section: Mean
        fullLocal.insertAdjacentHTML('beforeend', '<li><strong>Mean Values:</strong></li>');
        [
          { label: 'Mean Normalised Strength (n-set): f<sub>cm,n,is</sub> = ', key: 'f<sub>cm,n,is</sub>' },
          { label: 'Mean Prediction (m-set): f<sub>cm,m,is</sub> = ',       key: 'f<sub>cm,m,is</sub>' },
        ].forEach(item => {
          let v = localDetails[item.key];
          fullLocal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(2)} MPa</li>`
          );
        });
        // Section: Std Dev
        fullLocal.insertAdjacentHTML('beforeend', '<li><strong>Standard Deviations:</strong></li>');
        [
          { label: 'Test Variation: s<sub>fc,is,test</sub> = ',     key: 's<sub>fc,is,test</sub>' },
          { label: 'Model Variation: s<sub>θ,test</sub> = ',        key: 's<sub>theta,test</sub>' },
          { label: 'Overall: s<sub>fc,is</sub> = ',                 key: 's<sub>fc,is</sub>' },
        ].forEach(item => {
          let v = localDetails[item.key];
          fullLocal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(3)} MPa</li>`
          );
        });
        // Section: CoV
        fullLocal.insertAdjacentHTML('beforeend', '<li><strong>Coefficient of Variation:</strong></li>');
        [
          { label: 'CoV (lognormal): V<sub>fc,is</sub> = ',               key: 'V<sub>fc</sub>' },
          { label: 'CoV (logStudent-t): V<sub>fc,is,corr</sub> = ',   key: 'V<sub>fc,is,corr</sub>' },
        ].forEach(item => {
          let v = localDetails[item.key];
          fullLocal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(4)}</li>`
          );
        });
        // Section: Sample Size
        fullLocal.insertAdjacentHTML('beforeend', '<li><strong>Sample Size Effect:</strong></li>');
        [
          { label: 'DoF: n<sub>eff</sub> = ',                          key: 'n<sub>eff</sub>' },
          { label: 'Sample Size Coefficient: k<sub>n,eff</sub> = ',    key: 'k<sub>n,eff</sub>' },
        ].forEach(item => {
          let v = localDetails[item.key];
          fullLocal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(3)}</li>`
          );
        });
        // Section: Characteristic Value
        fullLocal.insertAdjacentHTML('beforeend', '<li><strong>Characteristic Value:</strong></li>');
        [
          { label: 'Lognormal Distribution: f<sub>ck,is</sub> = ',     key: 'f<sub>ck,is</sub>' },
        ].forEach(item => {
          let v = localDetails[item.key];
          fullLocal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(2)} MPa</li>`
          );
        });
        // Section: Equation Parameters
        fullLocal.insertAdjacentHTML('beforeend', '<li><strong>Curve Parameters:</strong></li>');
        [
          { label: 'Slope = ',         key: 'Slope' },
          { label: 'Intercept = ',     key: 'Intercept' },
          { label: 'R<sup>2</sup> = ', key: 'R^2' },
        ].forEach(item => {
          let v = localDetails[item.key];
          fullLocal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(6)}</li>`
          );
        });

        // Global Tab (Full Details)
        fullGlobal.innerHTML = '';
        // Section: Predictions
        const predsSection = `
          <li><strong>Calibration Set Predictions:</strong></li>
          <li>f<sub>c,n,is,pred</sub> (MPa) = [
              ${(json.global_details.n_pred  || [])
                  .map(v => v.toFixed(2))        
                  .join(", ") }]</li>
          <li><strong>Complementary Set Predictions:</strong></li>
          <li>f<sub>c,m,is,pred</sub> (MPa) = [
              ${(json.global_details.m_pred  || [])
                  .map(v => v.toFixed(2))
                  .join(", ") }]</li>`;
        fullGlobal.insertAdjacentHTML("beforeend", predsSection);
        // Section: Mean
        fullGlobal.insertAdjacentHTML('beforeend', '<li><strong>Mean Values:</strong></li>');
        [
          { label: 'Mean Normalised Strength (n-set): f<sub>cm,n,is</sub> = ', key: 'f<sub>cm,n,is</sub>' },
          { label: 'Mean Prediction (m-set): f<sub>cm,m,is</sub> = ',       key: 'f<sub>cm,m,is</sub>' },
        ].forEach(item => {
          let v = globalDetails[item.key];
          fullGlobal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(2)} MPa</li>`
          );
        });
        // Section: Std Dev
        fullGlobal.insertAdjacentHTML('beforeend', '<li><strong>Standard Deviations:</strong></li>');
        [
          { label: 'Test Variation: s<sub>fc,is,test</sub> = ',     key: 's<sub>fc,is,test</sub>' },
          { label: 'Model Variation: s<sub>θ,test</sub> = ',        key: 's<sub>theta,test</sub>' },
          { label: 'Overall: s<sub>fc,is</sub> = ',                 key: 's<sub>fc,is</sub>' },
        ].forEach(item => {
          let v = globalDetails[item.key];
          fullGlobal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(3)} MPa</li>`
          );
        });
        // Section: CoV
        fullGlobal.insertAdjacentHTML('beforeend', '<li><strong>Coefficient of Variation:</strong></li>');
        [
          { label: 'CoV (lognormal): V<sub>fc,is</sub> = ',               key: 'V<sub>fc</sub>' },
          { label: 'CoV (logStudent-t): V<sub>fc,is,corr</sub> = ',   key: 'V<sub>fc,is,corr</sub>' },
        ].forEach(item => {
          let v = globalDetails[item.key];
          fullGlobal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(4)}</li>`
          );
        });
        // Section: Sample Size
        fullGlobal.insertAdjacentHTML('beforeend', '<li><strong>Sample Size Effect:</strong></li>');
        [
          { label: 'DoF: n<sub>eff</sub> = ',                          key: 'n<sub>eff</sub>' },
          { label: 'Sample Size Coefficient: k<sub>n,eff</sub> = ',    key: 'k<sub>n,eff</sub>' },
        ].forEach(item => {
          let v = globalDetails[item.key];
          fullGlobal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(3)}</li>`
          );
        });
        // Section: Characteristic Value
        fullGlobal.insertAdjacentHTML('beforeend', '<li><strong>Characteristic Value:</strong></li>');
        [
          { label: 'Lognormal Distribution: f<sub>ck,is</sub> = ',     key: 'f<sub>ck,is</sub>' },
        ].forEach(item => {
          let v = globalDetails[item.key];
          fullGlobal.insertAdjacentHTML('beforeend',
            `<li>${item.label} ${parseFloat(v).toFixed(2)} MPa</li>`
          );
        });


        const chartMap = {};
        //  Helper to destroy charts
        function destroyIfExists(id) {
          const canvas = document.getElementById(id);
          const existing = Chart.getChart(canvas);
          if (existing) {
            existing.destroy();
          }
        }
        // Helper function to draw the local regression scatter + line with Chart.js
        function drawScatter(id, xData, yData, xLabel, ylabel, slope, intercept) {
          // destroy old if present
          destroyIfExists(id);

          const ctx = document.getElementById(id).getContext('2d');
          chartMap[id] = new Chart(document.getElementById(id), {
            type: 'scatter',
            data: {
              datasets: [{
                label: '',
                data: xData.map((x,i)=>({x, y: yData[i]})),
                pointBackgroundColor: '#662D91', 
                showLine: false,
                borderColor: 'black',
                borderWidth: 1,
                pointRadius: 4,
                tension: 0
              },
              {
                label: 'Fit',
                type: 'line',
                data: (() => {
                  const minX = Math.min(...xData), maxX = Math.max(...xData);
                  return [
                    { x: minX, y: slope * minX + intercept },
                    { x: maxX, y: slope * maxX + intercept }
                  ];
                })(),
                borderColor: 'black',
                borderWidth: 1,
                fill: false,
                pointRadius: 0
              }
            ]
          },
            options: {
              maintainAspectRatio: false, // Fill the container
              scales: {
                x: { title: { display: true, text: xLabel, font: { size: 14 } } },
                y: { title: { display: true, text: ylabel, font: { size: 14 } } }
              },
              plugins: {
                legend: { display: false } },
              layout: { padding: { top: 10, bottom: 30 } },

              // draw the equation below the x-axis
              plugins: [{
                id: 'drawEquation',
                afterDraw(chart) {
                  const {ctx, chartArea: {left, right, bottom}} = chart;
                  const xVar = xLabel.match(/[A-Za-zₚ]+/)[0];
                  const eqText = `${ylabel} = ${slope.toFixed(3)}·${xVar} + ${intercept.toFixed(3)}`;
                  ctx.save();
                  ctx.font = 'bold 12px Arial';
                  ctx.fillStyle = '#000';
                  ctx.textAlign = 'center';
                  ctx.fillText(eqText, (left + right) / 2, bottom + 20);
                  ctx.restore();
                }
              }]
            }
          });
        }

        // Helper like drawScatter, but taking two datasets
        function drawGlobal(id, x1,y1, x2,y2, xLabel, yLabel) {
          // destroy old if present
          destroyIfExists(id);

          const ctx = document.getElementById(id).getContext('2d');
          chartMap[id] = new Chart(ctx, {
            type: 'scatter',
            data: {
              datasets: [
                {
                  label: 'Global',
                  data: x1.map((x,i)=>({x, y:y1[i]})),
                  pointBackgroundColor: '#662D91', 
                  pointRadius: 3,
                  order: 1
                },
                {
                  label: 'You',
                  data: x2.map((x,i)=>({x, y:y2[i]})),
                  pointBackgroundColor: '#72A64B',
                  borderColor: 'black',
                  pointRadius: 4,
                  order: 2
                }
              ]
            },
            options: {
              maintainAspectRatio: false,
              scales: {
                x: { title: { display: true, text: xLabel } },
                y: { title: { display: true, text: yLabel } }
              },
              plugins: { legend: { position:'top' } },
              layout: { padding: 10 }
            },
            plugins:[{
              id: 'drawYouOnTop',
              afterDatasetsDraw(chart) {
                const you = chart.data.datasets[1];
                const xScale = chart.scales.x, yScale = chart.scales.y;
                const ctx = chart.ctx;
                ctx.save();
                you.data.forEach(pt=>{
                  const px = xScale.getPixelForValue(pt.x),
                        py = yScale.getPixelForValue(pt.y);
                  ctx.beginPath();
                  ctx.arc(px, py, you.pointRadius || 6, 0, Math.PI*2);
                  ctx.fillStyle = you.pointBackgroundColor;
                  ctx.fill();
                  if (you.borderWidth) {
                    ctx.lineWidth = you.borderWidth;
                    ctx.strokeStyle = you.borderColor;
                    ctx.stroke();
                  }
                });
                ctx.restore();
              }
            }]
          });
        }

        // Show the Bootstrap modal
        const modalEl = document.getElementById("analysisModal");
        const analysisModal = new bootstrap.Modal(modalEl);
        const tabList = document.getElementById("analysisTabs");
        
        /* Force the Summary tab every time the modal is *about to* show */
        modalEl.addEventListener('show.bs.modal', () => {
          const summaryBtn = document.querySelector(
                '#analysisTabs button[data-bs-target="#summaryPane"]');
          if (summaryBtn) {
            bootstrap.Tab.getOrCreateInstance(summaryBtn).show();
          }
        });
        
        analysisModal.show();

        // Listen for local tab to finish showing
        tabList.addEventListener("shown.bs.tab", e => {
          if (e.target.dataset.bsTarget !== "#localPane") return;  // only run for the Local Calibration tab
          const json        = window._lastAnalysisJSON;
          const local       = json.local_details;
          const local_plots = json.local_plots;
          const rawSlope    = local.Slope;                            // full precision
          const intercept   = local.Intercept;          
          const selectedNDT = document.getElementById("combined_ndt").value;
          const xVar        = selectedNDT === "upv" ? 'Vₚ' : 'RN';
          const xVar1       = 'Vₚ', xVar2 = 'RN';
          const yLabel      = 'f₍c,cyl₎ (MPa)';
          const Xn_Vp       = local_plots.Xn_Vp  || [];
          const Xn_RN       = local_plots.Xn_RN  || [];
          const Y           = local_plots.Y      || [];

          let displaySlope;
          if (Array.isArray(rawSlope)) {
            displaySlope = rawSlope.map(m => m.toFixed(3));
          } else {
            displaySlope = rawSlope.toFixed(3);
          }

          // Grab the chart-container wrappers
          const cont1 = document.querySelector("#local-plot-1").closest(".chart-container");
          const cont2 = document.querySelector("#local-plot-2").closest(".chart-container");

          // First hide both
          cont1.style.display = "none";
          cont2.style.display = "none";

          if (!Y.length) return;

          // clear previous drawings, if any
          document.getElementById("local-plot-1").getContext('2d').clearRect(0,0,600,350);
          document.getElementById("local-plot-2").getContext('2d').clearRect(0,0,600,350);

          if (selectedNDT === "upv"    && Xn_Vp.length) {
            cont1.style.display = "block";
            drawScatter("local-plot-1", Xn_Vp, Y, "Vₚ (m/s)", yLabel, rawSlope, intercept);
          }
          else if (selectedNDT === "rh" && Xn_RN.length) {
            cont1.style.display = "block";
            drawScatter("local-plot-1", Xn_RN, Y, "Rebound Number, RN", yLabel, rawSlope, intercept);
          }
          else if (selectedNDT === "sonreb" && Xn_Vp.length && Xn_RN.length) {
            cont1.style.display = "block";
            cont2.style.display = "block";
            drawScatter("local-plot-1", Xn_Vp, Y, "Vₚ (m/s)", yLabel, rawSlope, intercept);
            drawScatter("local-plot-2", Xn_RN, Y, "Rebound Number, RN", yLabel, rawSlope, intercept);
          }

          if (selectedNDT==='sonreb') {
            document.getElementById("local-reg-eq").innerHTML =
              `<strong>Multivariate Regression Equation:</strong><br>
               f<sub>c,cyl</sub> = ${displaySlope[0]}·${xVar1} + ${displaySlope[1]}·${xVar2} + ${intercept.toFixed(3)}`;
          }
          else {
            // univariate
            document.getElementById("local-reg-eq").innerHTML =
              `<strong>Regression Equation:</strong><br>
               f<sub>c,cyl</sub> = ${displaySlope}·${xVar} + ${intercept.toFixed(3)}`;
          }

        });
        
        // Listen for global to finish showing
        tabList.addEventListener("shown.bs.tab", e => {
          if (e.target.dataset.bsTarget === "#globalPane") {
            const json               = window._lastAnalysisJSON;
            const global_plots       = json.global_plots;
            const userPlots          = json.local_plots;    // user's points are in local_plots too
            const selectedNDT        = document.getElementById("combined_ndt").value;
            const yLabel = "f₍c,cyl₎ (MPa)";
            const xLabel1 = "Vₚ (m/s)";
            const xLabel2 = "Rebound Number, RN";

            const c1 = document.getElementById("global-plot-1"),
                  c2 = document.getElementById("global-plot-2");

            // hide both to start
            c1.style.display = "none";
            c2.style.display = "none";

            if (selectedNDT === "sonreb") {
            // Two plots
            c1.style.display = "block";
            drawGlobal("global-plot-1", global_plots.X_global_Vp, global_plots.Y_global, userPlots.Xn_Vp, userPlots.Y, xLabel1, yLabel );

            c2.style.display = "block";
            drawGlobal("global-plot-2", global_plots.X_global_RN, global_plots.Y_global, userPlots.Xn_RN, userPlots.Y, xLabel2, yLabel );
            } else {
              // One plot
              const xG = global_plots.X_global, xU = selectedNDT==="upv" ? userPlots.Xn_Vp : userPlots.Xn_RN;
              const xLab = selectedNDT==="upv" ? xLabel1 : xLabel2;
              c1.style.display = "block";
              drawGlobal("global-plot-1", xG, global_plots.Y_global, xU, userPlots.Y, xLab, yLabel);
            }
          }
        });
      })

      .catch(err => console.error("🔥 fetch error:", err))
    };


  // Initialize all [data-bs-toggle="tooltip"] elements
  // var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  // tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));

  window.highlightEmptyRows = highlightEmptyRows;

});


//  Function defining the minimum number of required rows in each table
function countFilledRows(tbody, colsNeeded) {
  // a row is "filled" when every <input> in the row in colsNeeded has a non-blank value
  return Array.from(tbody.querySelectorAll('tr')).filter(tr => {
    const inputs = Array.from(tr.querySelectorAll('input')).filter(inp =>
      !colsNeeded.length || colsNeeded.includes(inp.name));
    if (!inputs.length) return false;
    return inputs.every(inp => inp.value.trim() !== '');
  }).length;
}

// Customise the table error messages
function showTableError(containerId, msg) {
  let div = document.querySelector(`#${containerId} .error-message`);
  if (!div) {
    div = document.createElement('div');
    div.className = 'error-message text-danger mt-2';
    document.getElementById(containerId).appendChild(div);
  }
  div.innerText = msg;
  div.style.display = 'block';
}

function validateCombined() {
    let isValid = true;
    const form = document.getElementById("combined-form");

    // Clear all previous errors
    form.querySelectorAll(".error-message").forEach(div => {
        div.innerText = "";
        div.style.display = "none";
    });
    form.querySelectorAll(".is-invalid").forEach(el => {
        el.classList.remove("is-invalid");
    });

    // 2) Validate each visible, required field
    const requiredFields = form.querySelectorAll("input[required], select[required]");
    requiredFields.forEach(field => {
        // skip hidden or disabled
        if (field.disabled || field.offsetParent === null) return;

        // built-in validity check
        if (!field.checkValidity()) {
          // try to find the companion <div>
          let errDiv = field.parentElement.querySelector('.error-message');
          if (!errDiv) {                          // create if absent
            errDiv = document.createElement('div');
            errDiv.className = 'error-message text-danger';
            field.parentElement.appendChild(errDiv);
          }
          errDiv.innerText     = field.validationMessage;
          errDiv.style.display = 'block';
          field.classList.add('is-invalid');
          isValid = false;
        }
    });

    // Custom 'other' checks
    const upvSelect  = document.getElementById('upv_standard');
    const upvCustom  = document.getElementById('comb-custom-upv'); 
    if (upvSelect?.value === 'other_upv') {
      if (!upvCustom.value.trim()) {
        let div = upvCustom.parentElement.querySelector('.error-message');
        if (!div) {                           // create if template missing
          div = document.createElement('div');
          div.className = 'error-message text-danger';
          upvCustom.parentElement.appendChild(div);
        }
        div.innerText     = 'Please enter the custom UPV Standard.';
        div.style.display = 'block';
        upvCustom.classList.add('is-invalid');
        isValid = false;
      }
    }

    const rhSelect  = document.getElementById('rh_standard');
    const rhCustom  = document.getElementById('comb-custom-rh'); 
    if (rhSelect?.value === 'other_rh') {
      if (!rhCustom.value.trim()) {
        let div = rhCustom.parentElement.querySelector('.error-message');
        if (!div) {                           // create if template missing
          div = document.createElement('div');
          div.className = 'error-message text-danger';
          rhCustom.parentElement.appendChild(div);
        }
        div.innerText     = 'Please enter the custom RH Standard.';
        div.style.display = 'block';
        rhCustom.classList.add('is-invalid');
        isValid = false;
      }
    }

    // Focus the first invalid field, and alert once if any failed
    if (!isValid) {
        const firstErrorEl = form.querySelector(".is-invalid");
        firstErrorEl?.focus();
        alert("Please correct the highlighted fields.");
    }

    // Validate the tables
    const calibBody      = document.getElementById("calib-rows");
    const compBody       = document.getElementById("comp-rows");
    const calibFilled = countFilledRows(document.getElementById('calib-rows'),
                          ['velocity_n[]','rebound_number_n[]','width_diameter_n[]','height_n[]','fc_n[]']);
    const compFilled  = countFilledRows(document.getElementById('comp-rows'),
                          ['velocity_m[]','rebound_number_m[]']);
    const blankN = highlightEmptyRows(calibBody, 8, true);
    const blankM = highlightEmptyRows(compBody , 3 , true);

    if (calibFilled < 8) {
      showTableError('calibration-set-container',
        'You must fill in a minimum of 8 calibration tests');
      isValid = false;
    }
    if (compFilled < 3) {
      showTableError('complementary-container',
        'You must perform a minimum of 3 additional NDT tests for the statistical analysis to be valid');
      isValid = false;
    }
    if (blankN || blankM) {
      alert('Some added rows are empty. Please complete them or delete them.');
      isValid = false;
    }

    // console.log(`Validation result: ${isValid ? "✅ Passed" : "❌ Failed"}`);
    console.log('validateCombined result →', isValid);

    return isValid;
}
