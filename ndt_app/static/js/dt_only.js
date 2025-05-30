document.addEventListener("DOMContentLoaded", function() {
  const headersRow = document.getElementById("dt-headers");
  const body       = document.getElementById("dt-rows");
  const addBtn     = document.getElementById("add-dt-row");
  const container  = document.getElementById("dt-set-container");
  const summaryUl  = document.getElementById("dt-summary");

  const dtOnlyform = document.getElementById('dtonly-form');
  dtOnlyform.addEventListener('submit', e => { e.preventDefault(); if (!validateDtOnly())  return; analyseDTOnly(); }, true); 

  // Helper to build headers & rows
  function buildTable() {
    headersRow.innerHTML = [
      {class:'w-idx', text:''},
      {text:'Core Diameter (mm)'},
      {text:'Core Height (mm)'},
      {text:'Raw f<sub>c,core</sub> (MPa)'},
      {class:'w-del', text:''}
    ].map(obj => {
      const cls = obj.class ? ` class="${obj.class}"` : '';
      return `<th${cls}>${obj.text}</th>`;
    }).join('');
    if (!body.children.length) { for (let i = 0; i < 8; i++) addRow(); }
    container.style.display = "";
  }

  // Add one new DT row
  function addRow() {
    const idx = body.children.length + 1;
    let html  = `<th class="w-idx pe-1">${idx}</th>
                <td><input type="number" class="form-control" name="diam[]"   min="1" required></td>
                <td><input type="number" class="form-control" name="height[]" min="1" required></td>
                <td><input type="number" class="form-control" name="fc[]"     min="0" step="any" required></td>`;

    if (idx > MIN_DT_ROWS) {
      html += `<td class="w-del">
                <button type="button" class="delete-row btn p-0 text-dark">
                  <i class="bi bi-dash-circle-fill"></i>
                </button>
              </td>`;
    } else {
      html += '<td class="w-del"></td>';
    }

    const tr = document.createElement('tr');
    tr.innerHTML = html;
    body.appendChild(tr);
  }

  body.addEventListener('click', e => {
    if (!e.target.closest('.delete-row')) return;
    e.target.closest('tr').remove();
    renumberDTRows();
  });

  // Function highlighting empty incomplete table rows 
    const MIN_DT_ROWS = 8;

  /* keep the numbers correct after deletes */
  function renumberDTRows() {
    Array.from(body.rows).forEach((tr,i) => {
      const cell = tr.querySelector('.w-idx');
      if (cell) cell.innerText = i + 1;
    });
  }

  /* same badge logic you now use in Combined */
  function markBlankDtRows(showMsg=false) {
    let anyBlank = false;
    Array.from(body.rows).forEach((tr,idx) => {
      const blank = [...tr.querySelectorAll('input')]
                      .every(inp => inp.value.trim()==='');
      const needs = blank && idx >= MIN_DT_ROWS;
      anyBlank ||= needs;

      tr.classList.toggle('table-danger', needs && showMsg);

      let badge = tr.querySelector('.badge-row-error');
      if (needs && showMsg) {
        if (!badge) {
          const delCell = tr.querySelector('.w-del') || tr.lastElementChild;
          badge = document.createElement('span');
          badge.className = 'badge-row-error';
          badge.dataset.bsToggle = 'tooltip';
          badge.title = 'Row is empty – fill or delete';
          delCell.appendChild(badge);
          bootstrap.Tooltip.getOrCreateInstance(badge);
        }
      } else if (badge) {
        bootstrap.Tooltip.getInstance(badge).dispose();
        badge.remove();
      }
    });
    return anyBlank;
  }

  addBtn.addEventListener('click', () => {
    addRow();
    markBlankDtRows(false);
  });

  // also run once after every manual edit:
  // document.querySelectorAll('#dt-rows')
  //   .forEach(tbody => tbody.addEventListener('input', () => markBlankDtRows(tbody, 8)));
  body.addEventListener('input', () => markBlankDtRows(false));

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

  // Excel-paste helper (only on this tbody)
  installExcelPaste("#dt-rows", addRow);

  // On show: build initial table
  buildTable();

  // Clear DT-Only table
  document.getElementById("dt-clear-btn").addEventListener("click", () => {
    const rows = document.querySelectorAll("#dt-rows tr");
    rows.forEach((tr, i) => {
      // Only keep the first 8 empty rows
      if (i < 8) {
        tr.querySelectorAll("input").forEach(inp => inp.value = "");
      } else {
        tr.remove();
      }
    });
  });


  // 5) AJAX submit
  window.analyseDTOnly = () => {
    const data = new FormData(document.getElementById("dtonly-form"));
    const payload = {};
    for (let [k, v] of data.entries()) {
      payload[k] = payload[k]||[];
      payload[k].push(v);
    }

    fetch("/analyse_dt_only/", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    })
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(json => {
      // populate modal
      const summaryUl = document.getElementById("dt-summary");
      const dtSummary = json.dt_summary || {};
      summaryUl.innerHTML = "";

      // Section: Mean
      summaryUl.insertAdjacentHTML('beforeend', '<li class="mt-4"><strong>Mean:</strong></li>');
      [
        { label: 'Mean Normalised Strength: f<sub>cm,is</sub> = ', key: 'Mean fc (MPa)' },
      ].forEach(item => {
        let v = dtSummary[item.key];
        summaryUl.insertAdjacentHTML('beforeend',
          `<li>${item.label} ${parseFloat(v).toFixed(2)} MPa</li>`
        );
      });
      // Section: Std Dev
      summaryUl.insertAdjacentHTML('beforeend', '<li class="mt-4"><strong>Standard Deviation:</strong></li>');
      [
        { label: 'Overall: s<sub>fc,is</sub> = ',              key: 'Std Dev (MPa)' },
      ].forEach(item => {
        let v = dtSummary[item.key];
        summaryUl.insertAdjacentHTML('beforeend',
          `<li>${item.label} ${parseFloat(v).toFixed(3)} MPa</li>`
        );
      });
      // Section: Sample Size
      summaryUl.insertAdjacentHTML('beforeend', '<li class="mt-4"><strong>Sample Size Effect:</strong></li>');
      [
        { label: 'Sample Size: n = ',                          key: 'n' },
        { label: 'Sample Size Coefficient: k<sub>n</sub> = ',  key: 'k_n' },
      ].forEach(item => {
        let v = dtSummary[item.key];
        summaryUl.insertAdjacentHTML('beforeend',
          `<li>${item.label} ${parseFloat(v).toFixed(3)}</li>`
        );
      });

      // Section: CoV
      summaryUl.insertAdjacentHTML('beforeend', '<li class="mt-4"><strong>Coefficient of Variation:</strong></li>');
      [
        { label: 'CoV (lognormal): V<sub>fc,is</sub> = ',           key: 'V_fc,is' },
        { label: 'CoV (logStudent-t): V<sub>fc,is,corr</sub> = ',   key: 'V_fc,is,corr' },
      ].forEach(item => {
        let v = dtSummary[item.key];
        summaryUl.insertAdjacentHTML('beforeend',
          `<li>${item.label} ${parseFloat(v).toFixed(4)}</li>`
        );
      });
      // Section: Characteristic Value
      summaryUl.insertAdjacentHTML('beforeend', '<li class="mt-4"><strong>Characteristic Strength:</strong></li>');
      [
        { label: 'f<sub>ck,is</sub> = ',     key: 'Characteristic fck' },
      ].forEach(item => {
        let v = dtSummary[item.key];
        summaryUl.insertAdjacentHTML('beforeend',
          `<li>${item.label} ${parseFloat(v).toFixed(2)} MPa</li>`
        );
      });
      // Section: Bias
      summaryUl.insertAdjacentHTML('beforeend', '<li class="mt-4"><strong>Bias Factor:</strong></li>');
      [
        { label: 'μ<sub>fc,is</sub> = ',     key: 'Bias' },
      ].forEach(item => {
        let v = dtSummary[item.key];
        summaryUl.insertAdjacentHTML('beforeend',
          `<li>${item.label} ${parseFloat(v).toFixed(3)}</li>`
        );
      });

      // Show the Bootstrap modal
      new bootstrap.Modal(document.getElementById("dtonlyModal")).show();
    })

    .catch(err => console.error("DT-Only error:",err));

  };

  // Initialize all [data-bs-toggle="tooltip"] elements
  // var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  // tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));

  window.markBlankDtRows = markBlankDtRows;

});


//  Function defining the minimum number of required rows in each table
function countFilledDtRows(tbody, colsNeeded) {
  // a row is "filled" when every <input> in the row in colsNeeded has a non-blank value
  return Array.from(tbody.querySelectorAll('tr')).filter(tr => {
    const inputs = Array.from(tr.querySelectorAll('input')).filter(inp =>
      !colsNeeded.length || colsNeeded.includes(inp.name));
    if (!inputs.length) return false;
    return inputs.every(inp => inp.value.trim() !== '');
  }).length;
}

// Customise the table error messages
function showDtTableError(containerId, msg) {
  const container = document.getElementById(containerId);
  const host = container.closest('.table-responsive') || container;

  let div = host.querySelector('.error-message');
  if (!div) {
    div = document.createElement('div');
    div.className = 'error-message text-danger mt-2';
    host.after(div);                
  }
  div.innerText = msg;
  div.style.display = 'block';
}


function validateDtOnly() {
    let isValid = true;
    const form = document.getElementById("dtonly-form");

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
    const upvStandard = document.getElementById("upv_standard");
    const customUpv   = document.getElementById("custom-upv");
    if (upvStandard?.value === "other_upv" && !customUpv.value.trim()) {
        const d = customUpv.parentElement.querySelector(".error-message");
        d.innerText     = "Please enter the custom UPV Standard.";
        d.style.display = "block";
        customUpv.classList.add("is-invalid");
        isValid = false;
    }

    const rhStandard = document.getElementById("rh_standard");
    const customRh   = document.getElementById("custom-rh");
    if (rhStandard?.value === "other_rh" && !customRh.value.trim()) {
        const d = customRh.parentElement.querySelector(".error-message");
        d.innerText     = "Please enter the new RH Standard.";
        d.style.display = "block";
        customRh.classList.add("is-invalid");
        isValid = false;
    }

    // Focus the first invalid field, and alert once if any failed
    if (!isValid) {
        const firstErrorEl = form.querySelector(".is-invalid");
        firstErrorEl?.focus();
        alert("Please correct the highlighted fields.");
    }

    // Validate the table
    const dtTableFilled = countFilledDtRows(document.getElementById('dt-rows'),
                          ['diam[]','height[]','fc[]']);

    if (dtTableFilled < 8) {
      showDtTableError('dt-set-container',
        'You must fill in a minimum of 8 core tests');
      isValid = false;
    }
    if (markBlankDtRows(true)) {
      alert('Some added rows are empty. Please complete them or delete them.');
      isValid = false;
    }

    // console.log(`Validation result: ${isValid ? "✅ Passed" : "❌ Failed"}`);
    console.log('validateDtOnly result →', isValid);

    return isValid;
}
