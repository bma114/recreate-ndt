document.addEventListener("DOMContentLoaded", function() {

    // To stay on the same tab at refresh - only need this once for the whole app
    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(el => {
        el.addEventListener('shown.bs.tab', e => {
        const target = e.target.getAttribute('data-bs-target')   // <button>
                    || e.target.getAttribute('href');             // <a>
        localStorage.setItem('ndt-last-tab', target);
        });
    });

    /* on load: restore if we have one */
    const saved = localStorage.getItem('ndt-last-tab');
    if (saved) {
        const trigger = document.querySelector(
            `[data-bs-target="${saved}"],[href="${saved}"]`);
        if (trigger) {
            bootstrap.Tab.getOrCreateInstance(trigger).show();
            // if the tab lives inside a dropdown, close the dropdown
            const ddToggle = trigger.closest('.dropdown')?.querySelector('.dropdown-toggle');
            if (ddToggle) bootstrap.Dropdown.getOrCreateInstance(ddToggle).hide();
        }
    }

    /* always start at top-override browser scroll restoration */
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    window.scrollTo({ top:0, left:0, behavior:'instant' });

    const ndtSelect           = document.getElementById("ndt");
    const featureCard         = document.getElementById("indiv-features-card");
    const upvBlock            = document.getElementById("indiv-upv-block");
    const rhBlock             = document.getElementById("indiv-rh-block");

    // UPV-specific fields for custom standards
    const upvSelect           = document.getElementById("upv_standard");
    const customUpvContainer  = document.getElementById("custom-upv-container");
    const customUpvInput      = document.getElementById("custom-upv");

    // RH-specific fields for custom standards
    const rhSelect            = document.getElementById("rh_standard");
    const customRhContainer   = document.getElementById("custom-rh-container");
    const customRhInput       = document.getElementById("custom-rh");

    const form = document.getElementById('indiv-form');
    form.addEventListener('submit', e => { e.preventDefault(); if (!validateIndividual()) return; showPredictModal(); }, true); 

    // Initially hide all feature sections
    featureCard.style.display = "none";
    upvBlock.style.display    = "none";
    rhBlock.style.display     = "none";

    function updateForm() {
        const ndt = ndtSelect.value;
        const predictBtn = document.getElementById("predict-btn");

        // 1) Show/hide the whole card
        featureCard.style.display = ndt ? "block" : "none";

        // 2) Show/hide each sub-block
        upvBlock.style.display = (ndt === "upv" || ndt === "sonreb") ? "block" : "none";
        rhBlock.style.display  = (ndt === "rh"  || ndt === "sonreb") ? "block" : "none";

        // whenever the NDT dropdown changes, toggle the button
        ndtSelect.addEventListener("change", () => {
            predictBtn.style.display = ndtSelect.value ? "inline-block" : "none";
        });
        // run once on load in case you're reloading with a value
        predictBtn.style.display = ndtSelect.value ? "inline-block" : "none";

        // 3) For every data-ndt field in those blocks, disable+remove-required if hidden,
        //    or re-enable+restore-required if shown.
        document
        .querySelectorAll("#indiv-upv-block [data-ndt], #indiv-rh-block [data-ndt]")
        .forEach(el => {
            // visible if its block is shown
            const isUpv = el.closest("#indiv-upv-block");
            const isRh  = el.closest("#indiv-rh-block");
            const shown = (isUpv   && upvBlock.style.display==="block")
                        || (isRh    && rhBlock.style.display==="block");
            if (shown) {
            el.disabled = false;
            if (el.dataset.wasRequired) {
                el.setAttribute("required","");
                delete el.dataset.wasRequired;
            }
            } else {
            // remember if it was required
            if (el.hasAttribute("required")) el.dataset.wasRequired = "true";
            el.disabled = true;
            el.removeAttribute("required");
            el.classList.remove("is-invalid");
            const err = el.parentElement.querySelector(".error-message");
            if (err) err.style.display = "none";
            }
        });
    }

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

    // Run function when selection changes
    ndtSelect.addEventListener("change", updateForm);
    updateForm();  // run once on page load, in case you have a default

    // On valid submit, show modal with result + conditional text
    window.showPredictModal = function() {
        if (!validateIndividual()) return;
        // 1) grab the NDT selection
        const ndt = document.getElementById("ndt").value;

        // 2) collect all of your feature inputs under the card
        const featureEls = document.querySelectorAll("#indiv-features-card [name]");
        const features = {};
        featureEls.forEach(el => {
            // if it's disabled or hidden, we skip it
            if (el.disabled) return;
            const val = el.value.trim();    // remove surrounding spaces
            if (val === '') return;         // don't send blank entries
            features[el.name] = el.value;
        });

        // 3) build the exact payload shape
        const payload = { ndt, features };

        console.log("▶️ Predict payload:", payload);

        fetch("/predict/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        })
        .then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        })
        .then(json => {
            // show your modal & insert json.prediction
            document.getElementById("predict-result").innerHTML = json.prediction;

            // conditional text per NDT
            let info = "";
            if (ndt==="upv") {
                info = "UPV predictions have an RMSE of 4.20 MPa, reliable between 8.5 - 59.6 MPa.";
            } else if (ndt==="rh") {
                info = "RH predictions have an RMSE of 7.56 MPa, reliable between 11.7 - 85.0 MPa.";
            } else {
                info = "SonReb predictions have an RMSE of 3.19 MPa, reliable between 11.3 - 58.7 MPa.";
            }
            document.getElementById("predict-text").innerText = info;

            // Show modal
            new bootstrap.Modal(document.getElementById("predictModal")).show();
        })
        .catch(err => {
            console.error("Predict error:", err);
            alert("Sorry, something went wrong. Check the console for details.");
        });
    }

});


function validateIndividual() {
    let isValid = true;
    const form = document.getElementById("indiv-form");

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
        const errDiv = field.parentElement.querySelector(".error-message");
        errDiv.innerText = field.validationMessage;
        errDiv.style.display = "block";
        field.classList.add("is-invalid");
        isValid = false;
        }
    });

    // Custom 'other' checks
    const upvSelect  = document.getElementById('upv_standard');
    const upvCustom  = document.getElementById('custom-upv'); 
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
    const rhCustom  = document.getElementById('custom-rh'); 
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

    // console.log(`Validation result: ${isValid ? "✅ Passed" : "❌ Failed"}`);
    console.log('validateIndividual result →', isValid);

    return isValid;
}