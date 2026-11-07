/* ==========================================================================
   Shared form helpers: number parsing, paste cleaning, field validation,
   amber notes and the error summary. Used by every tab with a form.
   ========================================================================== */

const fmt = (x, d = 1) => Number(x).toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtPlain = x => Number(x).toLocaleString("en-GB", { maximumFractionDigits: 3 });
const isNum = el => el.dataset.type === "number";

/* Read a number typed or pasted in any common format:
   "4300", "4,300", "4 300", "4.300,5", "0,45", "52 kHz", "1.2e3". Returns NaN if it is not a number. */
function parseNum(raw) {
  let s = String(raw ?? "").trim().replace(/[  \s']/g, "");
  s = s.replace(/[a-zA-Z\/%°]+$/, "");                 // trailing unit, e.g. m/s, kHz, MPa
  if (!s) return NaN;
  const hasDot = s.includes("."), hasComma = s.includes(",");
  if (hasDot && hasComma) {
    // the later separator is the decimal mark
    s = s.lastIndexOf(",") > s.lastIndexOf(".") ? s.replace(/\./g, "").replace(",", ".") : s.replace(/,/g, "");
  } else if (hasComma) {
    s = /^-?\d{1,3}(,\d{3})+$/.test(s) ? s.replace(/,/g, "") : s.replace(",", ".");
  }
  return /^-?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(s) ? Number(s) : NaN;
}

/* Tidy one cell of pasted text (zero-width characters, repeated spaces) */
function tidyCell(text) {
  return String(text ?? "").replace(/[​﻿]/g, "").replace(/\s+/g, " ").trim();
}

/* Clean text pasted into a single box: first cell only, no line breaks or tabs */
function cleanPaste(text, numeric) {
  const cell = tidyCell(String(text).split(/\r?\n|\t/).map(c => c.trim()).find(c => c !== "") || "");
  if (!numeric) return cell;
  const x = parseNum(cell);
  return Number.isFinite(x) ? String(x) : cell;
}

/* Split pasted text into a grid of cells (Excel: tabs and new lines; Word/PDF: tabs, or runs of spaces) */
function pasteGrid(text) {
  const lines = String(text).replace(/\r/g, "").split("\n");
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  return lines.map(line => {
    const cells = line.includes("\t") ? line.split("\t") : line.trim().split(/\s{2,}|\s(?=-?\d)/);
    return cells.map(tidyCell);
  });
}

function labelHtml(label) {
  const c = label.cloneNode(true);
  c.querySelectorAll(".req, .opt").forEach(n => n.remove());
  return c.innerHTML.trim();
}
function labelText(label) {
  const c = label.cloneNode(true);
  c.querySelectorAll(".req, .opt").forEach(n => n.remove());
  return c.textContent.trim();
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* Out-of-range note for a number, or "" (the amber, non-blocking message) */
function rangeNote(x, lo, hi, unit, what = "The result") {
  if (!Number.isFinite(x)) return "";
  const below = lo !== undefined && lo !== null && x < Number(lo);
  const above = hi !== undefined && hi !== null && x > Number(hi);
  if (!below && !above) return "";
  const u = unit ? ` ${unit}` : "";
  const range = lo != null && hi != null ? `${fmtPlain(lo)}–${fmtPlain(hi)}${u}`
              : lo != null ? `above ${fmtPlain(lo)}${u}` : `up to ${fmtPlain(hi)}${u}`;
  return `Outside most of the training data (${range}). ${what} will be less reliable.`;
}

/* --------------------------------------------------------------------------
   Field kit: validation, amber notes, paste and live feedback for the
   label/input/feedback blocks (".field") inside `scope`.
   -------------------------------------------------------------------------- */
function createFieldKit({ form, scope, summary, noteWhat = "The prediction", onEdit = () => {} }) {
  const kit = { submitted: false };
  const inScope = el => scope.contains(el) && el.matches("input, select") && !el.closest(".no-kit");

  kit.controls = () => [...scope.querySelectorAll("input, select")]
    .filter(el => !el.disabled && !el.closest(".no-kit") && el.closest("[hidden]") === null);

  const labelOf = el => el.dataset.label || el.name;

  kit.errorFor = el => {
    const v = el.value.trim();
    const isSelect = el.tagName === "SELECT";
    if (!v) return !el.required ? "" : el.dataset.requiredMsg
      || (isSelect ? `Select the ${labelOf(el)}.` : `Enter the ${labelOf(el)}.`);
    if (isNum(el)) {
      const x = parseNum(v);
      if (!Number.isFinite(x)) return "Enter a number, for example 4300 or 0.45.";
      if (el.dataset.integer !== undefined && !Number.isInteger(x)) return "Enter a whole number.";
      const min = el.dataset.min !== undefined ? Number(el.dataset.min) : null;
      const max = el.dataset.max !== undefined ? Number(el.dataset.max) : null;
      const unit = el.dataset.unit ? ` ${el.dataset.unit}` : "";
      if ((min !== null && x < min) || (max !== null && x > max))
        return `Enter a value between ${fmtPlain(min)} and ${fmtPlain(max)}${unit}.`;
    } else if (el.dataset.pattern && !new RegExp(el.dataset.pattern).test(v)) {
      return `Use letters only for the ${labelOf(el)}, e.g. Netherlands.`;
    }
    return "";
  };

  // Non-blocking notes: values outside most of the training data, or unfamiliar names
  kit.noteFor = el => {
    const v = el.value.trim();
    if (!v) return "";
    if (isNum(el)) return rangeNote(parseNum(v), el.dataset.softMin, el.dataset.softMax, el.dataset.unit, noteWhat);
    if (el.dataset.known) {
      const list = document.getElementById(el.dataset.known);
      const known = list ? [...list.options].map(o => o.value.toLowerCase()) : [];
      if (!known.includes(v.toLowerCase().replace(/\s+/g, " ")))
        return `Not in the training database, so it will be treated as an unknown ${labelOf(el)}. Check the spelling against the suggestions.`;
    }
    return "";
  };

  kit.showField = el => {
    const msg = kit.errorFor(el);
    const err = document.getElementById(`${el.id}-error`);
    el.classList.toggle("is-invalid", !!msg);
    el.setAttribute("aria-invalid", msg ? "true" : "false");
    if (err) err.textContent = msg;
    const noteEl = el.closest(".field")?.querySelector(".field-note");
    if (noteEl && el.closest(".other-box") === null) {
      const note = msg ? "" : kit.noteFor(el);
      noteEl.innerHTML = note ? `<i class="bi bi-info-circle" aria-hidden="true"></i> ${escapeHtml(note)}` : "";
      noteEl.hidden = !note;
    }
    return msg;
  };

  kit.clearField = el => {
    el.classList.remove("is-invalid");
    el.removeAttribute("aria-invalid");
    delete el.dataset.touched;
    const err = document.getElementById(`${el.id}-error`);
    if (err) err.textContent = "";
    const noteEl = el.closest(".field")?.querySelector(".field-note");
    if (noteEl && el.closest(".other-box") === null) { noteEl.hidden = true; noteEl.innerHTML = ""; }
  };

  // "Enter a whole number." becomes "Concrete age: enter a whole number."
  kit.summaryMsg = (el, msg) => {
    if (/^(Enter the|Select )/.test(msg)) return msg;
    const label = el.closest(".other-box") ? el.closest(".other-box").querySelector("label")
                                            : el.closest(".field").querySelector("label");
    return `${labelText(label)}: ${msg.charAt(0).toLowerCase()}${msg.slice(1)}`;
  };

  kit.fieldProblems = (show = true) => kit.controls().reduce((acc, el) => {
    const msg = show ? kit.showField(el) : kit.errorFor(el);
    if (msg) acc.push({ id: el.id, focus: el, msg: kit.summaryMsg(el, msg) });
    return acc;
  }, []);

  kit.notes = () => [...scope.querySelectorAll(".field-note:not([hidden])")]
    .filter(n => n.closest("[hidden]") === null)
    .map(n => `${labelText(n.closest(".field").querySelector("label"))}: ${n.textContent.trim()}`);

  kit.renderSummary = problems => {
    if (!problems.length) { summary.hidden = true; return; }
    summary.querySelector("h2 span").textContent =
      problems.length === 1 ? "There is 1 problem to fix" : `There are ${problems.length} problems to fix`;
    const ul = summary.querySelector("ul");
    ul.innerHTML = "";
    problems.forEach(p => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `#${p.id}`; a.textContent = p.msg;
      a.addEventListener("click", ev => {
        ev.preventDefault();
        p.focus.focus({ preventScroll: true });
        p.focus.scrollIntoView({ block: "center", behavior: "smooth" });
      });
      li.appendChild(a); ul.appendChild(li);
    });
    summary.hidden = false;
  };

  kit.showSummary = problems => {
    kit.renderSummary(problems);
    if (problems.length) {
      summary.scrollIntoView({ block: "start", behavior: "smooth" });
      summary.focus({ preventScroll: true });
    }
  };

  // Called after edits once the summary is showing; the tab supplies the full problem list
  kit.refresh = () => {};

  /* "Other (type it in)" standard */
  kit.bindOther = (select, box, input, otherValue) => {
    const sync = () => {
      const useOther = !select.disabled && select.value === otherValue;
      box.hidden = !useOther;
      input.disabled = !useOther;
      input.required = useOther;
      if (!useOther) { input.value = ""; kit.clearField(input); }
    };
    select.addEventListener("change", () => { sync(); if (!box.hidden) input.focus(); });
    return sync;
  };

  /* paste into a single box */
  form.addEventListener("paste", e => {
    const el = e.target;
    if (!inScope(el) || el.type !== "text") return;
    const text = (e.clipboardData || window.clipboardData)?.getData("text");
    if (text == null) return;                 // let the browser handle it
    e.preventDefault();
    const clean = cleanPaste(text, isNum(el));
    const start = el.selectionStart ?? el.value.length, end = el.selectionEnd ?? el.value.length;
    el.value = (el.value.slice(0, start) + clean + el.value.slice(end)).slice(0, el.maxLength > 0 ? el.maxLength : undefined);
    el.dataset.touched = "1";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });

  // Typed numbers: tidy to a plain number when leaving the box, e.g. "0,45" becomes 0.45
  form.addEventListener("change", e => {
    const el = e.target;
    if (!inScope(el) || !isNum(el)) return;
    const x = parseNum(el.value);
    if (Number.isFinite(x) && String(x) !== el.value.trim()) el.value = String(x);
  }, true);

  // Live feedback: on leaving a field, then on every change once it has been checked
  form.addEventListener("focusout", e => {
    const el = e.target;
    if (!inScope(el) || el.disabled) return;
    if (el.value.trim() || kit.submitted) { el.dataset.touched = "1"; kit.showField(el); if (!summary.hidden) kit.refresh(); }
  });
  ["input", "change"].forEach(evt => form.addEventListener(evt, e => {
    const el = e.target;
    if (!inScope(el)) return;
    onEdit(el);
    if (el.dataset.touched || kit.submitted || el.tagName === "SELECT") { kit.showField(el); if (!summary.hidden) kit.refresh(); }
  }));

  return kit;
}

/* Spinner + label swap on a submit button */
function setButtonBusy(btn, busy, idleLabel, busyLabel) {
  btn.disabled = busy;
  btn.setAttribute("aria-busy", busy ? "true" : "false");
  const sp = btn.querySelector(".spinner-border");
  if (sp) sp.hidden = !busy;
  btn.querySelector(".btn-label").textContent = busy ? busyLabel : idleLabel;
  // a slow first request usually means the server is waking up: say so
  clearTimeout(btn._slowTimer);
  let note = btn.parentElement.querySelector(".busy-note");
  if (!busy) { note?.remove(); return; }
  btn._slowTimer = setTimeout(() => {
    if (!note) {
      note = document.createElement("p");
      note.className = "busy-note"; note.setAttribute("role", "status");
      btn.parentElement.appendChild(note);
    }
    note.textContent = "Still working. After a quiet period the server can take up to a minute to start.";
  }, 6000);
}

/* POST JSON and return the JSON reply; errors carry a plain-English message */
async function postJson(url, payload, timeoutMs = 120000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let r;
  try {
    r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify(payload), signal: ctrl.signal });
  } catch (err) {
    const e = new Error(err.name === "AbortError"
      ? "The server took too long to reply."
      : "The app could not reach the server. Check your internet connection.");
    e.retry = true;
    throw e;
  } finally {
    clearTimeout(timer);
  }
  const json = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = new Error(json.error || (r.status >= 500
      ? `The server ran into a problem (error ${r.status}).`
      : `The request was not accepted (error ${r.status}).`));
    e.retry = !json.error && r.status >= 500;
    throw e;
  }
  return json;
}

/* Show a failed request in the tab's error box */
function showServerError(box, lead, err) {
  box.innerHTML = `<strong>${lead}</strong> ${escapeHtml(err.message)}` +
    (err.retry ? " If the app has been idle, the server may still be starting up: wait a minute and try again." : "");
  box.hidden = false;
  box.scrollIntoView({ block: "center", behavior: "smooth" });
}

/* Copy text, then briefly confirm on the button */
async function copyWithFeedback(btn, text, idleHtml) {
  try {
    await navigator.clipboard.writeText(text);
    btn.innerHTML = '<i class="bi bi-check2" aria-hidden="true"></i> Copied';
  } catch (_) {
    btn.innerHTML = '<i class="bi bi-x" aria-hidden="true"></i> Copy not available';
  }
  setTimeout(() => { btn.innerHTML = idleHtml; }, 2000);
}

/* --------------------------------------------------------------------------
   Data table kit: spreadsheet-style entry with block paste from Excel,
   per-cell validation, amber notes and a running count of complete rows.

   columns: [{ key, name, label (html), short (plain text), unit,
               min, max, integer, soft(x) -> note text or "" }]
   -------------------------------------------------------------------------- */
let activeTable = null;
function createDataTable({ table, idPrefix, minRows, what, countEl, statusEl, errorsEl, notesEl, onEdit = () => {}, extraNotes = () => [] }) {
  const thead = table.querySelector("thead tr");
  const tbody = table.querySelector("tbody");
  const dt = { columns: [], submitted: false };
  let statusTimer = null;

  /* ---- undo history: the table keeps its own, so Ctrl+Z undoes pastes,
     row deletes and clears instead of jumping to another field ---- */
  const undoStack = [], redoStack = [];
  let typingRecorded = false;
  const snapshot = () => {
    const a = document.activeElement;
    const inp = a && tbody.contains(a) && a.matches("input.cell") ? a : null;
    return { rows: dt.fullData(), focus: inp ? { row: inp.closest("tr").sectionRowIndex, col: inp.dataset.col } : null };
  };
  const sameRows = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  function record() {
    const snap = snapshot();
    const top = undoStack[undoStack.length - 1];
    if (top && sameRows(top.rows, snap.rows)) return;
    undoStack.push(snap);
    if (undoStack.length > 100) undoStack.shift();
    redoStack.length = 0;
  }
  function restore(snap) {
    tbody.innerHTML = "";
    const n = Math.max(minRows, snap.rows.length);
    for (let i = 0; i < n; i++) dt.addRow(snap.rows[i] || {});
    tbody.querySelectorAll("input.cell").forEach(inp => { if (inp.value) inp.dataset.touched = "1"; });
    dt.validate(dt.submitted);
    onEdit();
    const f = snap.focus;
    const tr = f && tbody.rows[Math.min(f.row, tbody.rows.length - 1)];
    const target = (tr && tr.querySelector(`input.cell[data-col="${f.col}"]`)) || tbody.querySelector("input.cell");
    target?.focus();
  }
  function step(from, to, word) {
    if (!from.length) { flash(`Nothing to ${word.toLowerCase()} in this table.`); return; }
    typingRecorded = false;
    to.push(snapshot());
    restore(from.pop());
    flash(`${word} done.`, false);
  }
  dt.undo = () => step(undoStack, redoStack, "Undo");
  dt.redo = () => step(redoStack, undoStack, "Redo");
  dt.lastAction = 0;

  const cellLabel = c => c.short + (c.unit ? ` (${c.unit})` : "");

  function rowHtml(i) {
    const cells = dt.columns.map(c =>
      `<td><input type="text" class="form-control form-control-sm cell" data-col="${c.key}" id="${idPrefix}-r${i}-${c.key}"
         inputmode="decimal" autocomplete="off" maxlength="20" aria-label="Row ${i}, ${escapeHtml(cellLabel(c))}"></td>`).join("");
    return `<th scope="row" class="row-num">${i}</th>${cells}
      <td class="row-del"><button type="button" class="btn-row-del" aria-label="Delete row ${i}" title="Delete row">
        <i class="bi bi-x-lg" aria-hidden="true"></i></button></td>`;
  }

  function renumber() {
    [...tbody.rows].forEach((tr, k) => {
      const i = k + 1;
      tr.querySelector(".row-num").textContent = i;
      tr.querySelectorAll("input.cell").forEach(inp => {
        const c = dt.columns.find(col => col.key === inp.dataset.col);
        inp.id = `${idPrefix}-r${i}-${c.key}`;
        inp.setAttribute("aria-label", `Row ${i}, ${cellLabel(c)}`);
      });
      tr.querySelector(".btn-row-del").setAttribute("aria-label", `Delete row ${i}`);
    });
  }

  dt.addRow = (values = {}) => {
    const tr = document.createElement("tr");
    tr.innerHTML = rowHtml(tbody.rows.length + 1);
    tr.querySelectorAll("input.cell").forEach(inp => { inp.value = values[inp.dataset.col] ?? ""; });
    // values for columns not shown now (e.g. RN while UPV is selected) travel with the row
    tr._hidden = Object.fromEntries(Object.entries(values).filter(([k, v]) => v !== "" && v != null && !dt.columns.some(c => c.key === k)));
    tbody.appendChild(tr);
    return tr;
  };

  /* raw values per row, by column key */
  dt.data = () => [...tbody.rows].map(tr =>
    Object.fromEntries([...tr.querySelectorAll("input.cell")].map(inp => [inp.dataset.col, inp.value.trim()])));

  /* every value in each row, including columns hidden by the current method */
  dt.fullData = () => { const d = dt.data(); return [...tbody.rows].map((tr, k) => ({ ...(tr._hidden || {}), ...d[k] })); };
  dt.load = rows => {
    tbody.innerHTML = "";
    const n = Math.max(minRows, rows.length);
    for (let i = 0; i < n; i++) dt.addRow(rows[i] || {});
    tbody.querySelectorAll("input.cell").forEach(inp => { if (inp.value) inp.dataset.touched = "1"; });
    dt.validate(dt.submitted);
  };

  /* change columns (e.g. when the NDT method changes) and keep what was typed */
  dt.setColumns = cols => {
    const old = dt.fullData();
    dt.columns = cols;
    thead.innerHTML = `<th scope="col" class="row-num"><span class="visually-hidden">Row</span></th>` +
      cols.map(c => `<th scope="col">${c.label}${c.unit ? `<span class="col-unit">${c.unit}</span>` : ""}</th>`).join("") +
      `<th scope="col" class="row-del"><span class="visually-hidden">Delete</span></th>`;
    tbody.innerHTML = "";
    const n = Math.max(minRows, old.length);
    for (let i = 0; i < n; i++) dt.addRow(old[i] || {});
    dt.validate(false);
  };

  dt.reset = () => {
    dt.forgetHistory?.();
    tbody.innerHTML = "";
    for (let i = 0; i < minRows; i++) dt.addRow();
    dt.submitted = false;
    dt.validate(false);
    statusEl.textContent = "";
  };

  /* Validate. show=true marks cells and fills the error and note lists. */
  dt.validate = (show = true) => {
    const problems = [], notes = [], complete = [];
    let firstEmpty = null;
    [...tbody.rows].forEach((tr, k) => {
      const i = k + 1;
      const inputs = [...tr.querySelectorAll("input.cell")];
      const blank = inputs.every(inp => !inp.value.trim());
      const row = {};
      let ok = !blank;
      inputs.forEach(inp => {
        const c = dt.columns.find(col => col.key === inp.dataset.col);
        const v = inp.value.trim();
        let msg = "", note = "";
        if (!blank) {
          if (!v) msg = `Row ${i}: enter the ${c.short.toLowerCase()}, or clear the row.`;
          else {
            const x = parseNum(v);
            if (!Number.isFinite(x)) msg = `Row ${i}, ${c.short}: enter a number.`;
            else if (c.integer && !Number.isInteger(x)) msg = `Row ${i}, ${c.short}: enter a whole number.`;
            else if ((c.min != null && x < c.min) || (c.max != null && x > c.max))
              msg = `Row ${i}, ${c.short}: enter a value between ${fmtPlain(c.min)} and ${fmtPlain(c.max)}${c.unit ? " " + c.unit : ""}.`;
            else {
              row[c.key] = x;
              note = c.soft ? c.soft(x) : "";
            }
          }
        } else if (!firstEmpty) firstEmpty = inputs[0];
        if (msg) { ok = false; problems.push({ id: inp.id, focus: inp, msg }); }
        if (note) notes.push(`Row ${i}, ${c.short} ${fmtPlain(parseNum(v))}${c.unit ? " " + c.unit : ""}: ${note}`);
        if (show || !msg) {
          inp.classList.toggle("is-invalid", show && !!msg);
          inp.setAttribute("aria-invalid", show && msg ? "true" : "false");
          inp.title = show && msg ? msg : (note || "");
        }
        inp.classList.toggle("is-note", !!note && !msg);
      });
      if (ok) complete.push(Object.assign(row, { _row: i, _tr: tr }));
    });
    // notes that depend on other data (e.g. the calibration range)
    extraNotes(complete).forEach(({ row, key, text }) => {
      notes.push(text);
      const inp = row._tr.querySelector(`input.cell[data-col="${key}"]`);
      if (inp && !inp.classList.contains("is-invalid")) { inp.classList.add("is-note"); inp.title = text; }
    });

    const enough = complete.length >= minRows;
    if (!enough) {
      const focus = firstEmpty || tbody.querySelector("input.cell");
      problems.push({ id: focus?.id || table.id, focus, table: true,
        msg: `Enter at least ${minRows} complete ${what} (${complete.length} so far).` });
    }

    countEl.innerHTML = enough
      ? `<i class="bi bi-check-circle-fill" aria-hidden="true"></i> ${complete.length} complete ${what}`
      : `${complete.length} of at least ${minRows} ${what} complete`;
    countEl.classList.toggle("is-ok", enough);

    if (show) {
      const ul = errorsEl.querySelector("ul");
      const list = problems.map(p => p.msg);
      ul.innerHTML = list.slice(0, 8).map(m => `<li>${escapeHtml(m)}</li>`).join("") +
        (list.length > 8 ? `<li>and ${list.length - 8} more</li>` : "");
      errorsEl.hidden = !list.length;
    }
    const nl = notesEl.querySelector("ul");
    nl.innerHTML = notes.map(m => `<li>${escapeHtml(m)}</li>`).join("");
    notesEl.hidden = !notes.length;
    return { problems, notes, complete };
  };

  function flash(msg, offerUndo = false) {
    statusEl.innerHTML = escapeHtml(msg) + (offerUndo
      ? ` <button type="button" class="btn-undo" data-undo><i class="bi bi-arrow-counterclockwise" aria-hidden="true"></i> Undo</button>
         <span class="undo-key">or press ${/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl"}+Z</span>`
      : "");
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => { statusEl.textContent = ""; }, offerUndo ? 10000 : 4000);
  }
  statusEl.addEventListener("click", e => { if (e.target.closest("[data-undo]")) dt.undo(); });
  const touch = () => { dt.lastAction = Date.now(); activeTable = dt; };

  /* block paste from Excel / Word / PDF tables */
  tbody.addEventListener("paste", e => {
    const inp = e.target.closest("input.cell");
    if (!inp) return;
    const text = (e.clipboardData || window.clipboardData)?.getData("text");
    if (text == null) return;
    e.preventDefault();
    let grid = pasteGrid(text);
    if (!grid.length) return;
    record(); touch(); typingRecorded = false;
    // drop a header row copied with the data
    if (grid.length > 1 && grid[0].every(c => !Number.isFinite(parseNum(c)))) grid = grid.slice(1);

    const startRow = inp.closest("tr").sectionRowIndex;
    const cols = [...inp.closest("tr").querySelectorAll("input.cell")];
    const startCol = cols.indexOf(inp);
    let extra = 0;
    grid.forEach((cells, r) => {
      while (tbody.rows.length <= startRow + r) dt.addRow();
      const targets = [...tbody.rows[startRow + r].querySelectorAll("input.cell")];
      cells.forEach((val, c) => {
        const t = targets[startCol + c];
        if (!t) { extra = Math.max(extra, startCol + c - targets.length + 1); return; }
        const x = parseNum(val);
        t.value = Number.isFinite(x) ? String(x) : val;
        t.dataset.touched = "1";
      });
    });
    const width = Math.max(...grid.map(r => r.length));
    flash(`Pasted ${grid.length} row${grid.length > 1 ? "s" : ""} × ${Math.min(width, cols.length - startCol)} column${width > 1 ? "s" : ""}` +
          (extra ? `; ${extra} extra column${extra > 1 ? "s" : ""} ignored.` : "."), true);
    dt.validate(dt.submitted);
    onEdit();
  });

  tbody.addEventListener("beforeinput", e => {
    if (!e.target.matches("input.cell") || typingRecorded) return;
    record(); typingRecorded = true;
  });
  tbody.addEventListener("focusin", e => { if (e.target.matches("input.cell")) { typingRecorded = false; touch(); } });

  tbody.addEventListener("input", e => {
    if (!e.target.matches("input.cell")) return;
    e.target.dataset.touched = "1";
    dt.validate(dt.submitted);
    onEdit();
  });
  tbody.addEventListener("focusout", e => {
    const inp = e.target;
    if (!inp.matches("input.cell")) return;
    const x = parseNum(inp.value);
    if (Number.isFinite(x) && String(x) !== inp.value.trim()) inp.value = String(x);
    dt.validate(dt.submitted);
  });

  /* Enter moves down a row, like a spreadsheet */
  tbody.addEventListener("keydown", e => {
    if (e.key !== "Enter" || !e.target.matches("input.cell")) return;
    e.preventDefault();
    const tr = e.target.closest("tr");
    const col = e.target.dataset.col;
    const next = tr.nextElementSibling || dt.addRow();
    next.querySelector(`input.cell[data-col="${col}"]`).focus();
  });

  tbody.addEventListener("click", e => {
    const btn = e.target.closest(".btn-row-del");
    if (!btn) return;
    const tr = btn.closest("tr");
    const k = tr.sectionRowIndex;
    const hadData = [...tr.querySelectorAll("input.cell")].some(inp => inp.value.trim());
    record(); touch(); typingRecorded = false;
    tr.remove();
    while (tbody.rows.length < minRows) dt.addRow();  // keep the minimum number of rows, blank ones at the end
    renumber();
    dt.validate(dt.submitted);
    onEdit();
    // keep focus in the table so Ctrl+Z stays with it
    const next = tbody.rows[Math.min(k, tbody.rows.length - 1)];
    next?.querySelector(".btn-row-del")?.focus();
    if (hadData) flash(`Row ${k + 1} deleted.`, true);
  });

  dt.addBlankRow = () => {
    record(); touch();
    const tr = dt.addRow();
    tr.querySelector("input.cell").focus();
    dt.validate(dt.submitted);
  };

  dt.clear = () => {
    const hasData = dt.data().some(r => Object.values(r).some(v => v));
    if (!hasData) return;
    record(); touch(); typingRecorded = false;
    tbody.innerHTML = "";
    for (let i = 0; i < minRows; i++) dt.addRow();
    dt.validate(dt.submitted);
    onEdit();
    tbody.querySelector("input.cell")?.focus();
    flash("Table cleared.", true);
  };
  dt.forgetHistory = () => { undoStack.length = 0; redoStack.length = 0; };

  /* keyboard: Ctrl/Cmd+Z, Ctrl+Y, Ctrl/Cmd+Shift+Z while working in this table */
  const card = table.closest(".form-card") || table.parentElement;
  document.addEventListener("keydown", e => {
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    const k = e.key.toLowerCase();
    const isUndo = k === "z" && !e.shiftKey;
    const isRedo = k === "y" || (k === "z" && e.shiftKey);
    if (!isUndo && !isRedo) return;
    const a = document.activeElement;
    const inCard = a && card.contains(a);
    // focus left on the page itself just after a table action (e.g. a deleted row's button)
    const orphan = (!a || a === document.body) && activeTable === dt;
    if (!inCard && !orphan) return;
    e.preventDefault();
    isUndo ? dt.undo() : dt.redo();
  });

  return dt;
}


/* ==========================================================================
   Shared by the two test region tabs (combined DT + NDT, and DT only)
   ========================================================================== */
const fx = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "–" : fmt(v, d);
const pct = (v, d = 1) => fx(100 * v, d) + "%";

// Core table columns (hard limits block the analysis; soft limits give an amber note)
const CORE_COLS = {
  d:   { key: "d",   label: "Core diameter", short: "Core diameter", unit: "mm", min: 20, max: 300,
         soft: x => (x < 44 || x > 160) ? "outside the 44–160 mm range of the conversion factors; the factors for the nearest core size are used." : "" },
  h:   { key: "h",   label: "Core height", short: "Core height", unit: "mm", min: 20, max: 600 },
  fc:  { key: "fc",  label: "Core strength, f<sub>c,core</sub>", short: "Core strength", unit: "MPa", min: 1, max: 200 },
};

// Core conversion to the 150 x 300 mm cylinder, as returned by the server
function coreNormTable(rows) {
  return `<thead><tr><th class="col-idx">#</th><th>D (mm)</th><th>H (mm)</th><th>H/D</th><th>f<sub>c,core</sub> (MPa)</th>
    <th>f<sub>c,cyl</sub> (MPa)</th></tr></thead><tbody>` +
    rows.map((x, i) => `<tr${x.note ? ' class="is-note-row" title="' + escapeHtml(x.note) + '"' : ""}><td class="col-idx">${i + 1}</td><td>${fmtPlain(x.D)}</td>
      <td>${fmtPlain(x.H)}</td><td>${fx(x.H_over_D)}</td><td>${fx(x.fc_core)}</td>
      <td>${fx(x.fc_cyl)}</td></tr>`).join("") + "</tbody>";
}

// legend markers drawn at the same size as the points in the plot
function chartMarker(r, fill, stroke, bw) {
  const size = Math.ceil(2 * r + bw) + 1, dpr = 3;
  const c = document.createElement("canvas");
  c.width = c.height = size * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.beginPath(); ctx.arc(size / 2, size / 2, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill; ctx.fill();
  if (bw) { ctx.lineWidth = bw; ctx.strokeStyle = stroke; ctx.stroke(); }
  const img = new Image(size, size);
  img.src = c.toDataURL();
  return img;
}

// Legend that draws each dataset's own marker (set as dataset.legendMarker)
const legendWithMarkers = () => ({
  position: "top", align: "end",
  labels: { usePointStyle: true, boxWidth: 12, boxHeight: 7, font: { size: 11 },
    generateLabels: chart => Chart.defaults.plugins.legend.labels.generateLabels(chart).map(item => {
      const m = chart.data.datasets[item.datasetIndex].legendMarker;
      if (m) item.pointStyle = m;
      return item;
    }) },
});

/* ==========================================================================
   Drafts: keep what was entered in a tab through a page refresh or Back/Forward
   (sessionStorage: kept for this browser tab only, gone when it is closed)
   ========================================================================== */
function createDraft(key, { form, tables = {}, radioName = null }) {
  const storeKey = `ndt-draft-${key}`;
  const fields = () => [...form.querySelectorAll("input[id], select[id]")]
    .filter(el => !el.classList.contains("cell") && el.type !== "radio" && el.type !== "checkbox");
  const draft = {};

  draft.save = () => {
    const data = { radio: radioName ? (form.querySelector(`input[name="${radioName}"]:checked`)?.value || "") : "", fields: {}, tables: {} };
    fields().forEach(el => { if (el.value) data.fields[el.id] = el.value; });
    Object.entries(tables).forEach(([k, t]) => {
      const rows = t.fullData();
      while (rows.length && !Object.values(rows[rows.length - 1]).some(v => v)) rows.pop();
      if (rows.length) data.tables[k] = rows;
    });
    const empty = !data.radio && !Object.keys(data.fields).length && !Object.keys(data.tables).length;
    try { empty ? sessionStorage.removeItem(storeKey) : sessionStorage.setItem(storeKey, JSON.stringify(data)); } catch (_) { /* storage blocked */ }
  };
  let timer = null;
  draft.queue = () => { clearTimeout(timer); timer = setTimeout(draft.save, 300); };
  draft.clear = () => { clearTimeout(timer); try { sessionStorage.removeItem(storeKey); } catch (_) { /* storage blocked */ } };

  draft.restore = () => {
    let data = null;
    try { data = JSON.parse(sessionStorage.getItem(storeKey) || "null"); } catch (_) { data = null; }
    if (!data) return false;
    if (data.radio) {
      const r = form.querySelector(`input[name="${radioName}"][value="${data.radio}"]`);
      if (r) { r.checked = true; r.dispatchEvent(new Event("change", { bubbles: true })); }
    }
    const set = el => { if (el && data.fields[el.id] != null) el.value = data.fields[el.id]; };
    // selects first (they show or hide the "other" boxes), then the text fields
    fields().filter(el => el.tagName === "SELECT").forEach(el => {
      if (data.fields[el.id] == null) return;
      set(el); el.dispatchEvent(new Event("change", { bubbles: true }));
    });
    fields().filter(el => el.tagName === "INPUT").forEach(set);
    Object.entries(tables).forEach(([k, t]) => { if (data.tables[k]) t.load(data.tables[k]); });
    return true;
  };

  form.addEventListener("input", draft.queue);
  form.addEventListener("change", draft.queue);
  window.addEventListener("pagehide", draft.save);
  return draft;
}

/* After a result window closes, put keyboard focus back on the button that opened it */
function returnFocusOnClose(modalEl, btn) {
  modalEl.addEventListener("hidden.bs.modal", () => {
    const a = document.activeElement;
    if (a && a !== document.body && !modalEl.contains(a)) return;
    // the button can be hidden after "New analysis"; then start at the top of the form
    const target = btn.offsetParent ? btn : btn.form?.querySelector("input:not([disabled]), select:not([disabled])");
    target?.focus({ preventScroll: true });
  });
}

/* A result can arrive after the user has moved to another tab: go back to the
   tab that asked for it before showing the result window */
function showOwningTab(el) {
  const pane = el.closest(".tab-pane");
  if (!pane || pane.classList.contains("active")) return;
  const trigger = document.querySelector(`#mainTabs [href="#${pane.id}"]`);
  if (trigger) bootstrap.Tab.getOrCreateInstance(trigger).show();
}
