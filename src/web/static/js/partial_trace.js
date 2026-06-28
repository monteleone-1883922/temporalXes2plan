/**
 * partial_trace.js — Upload a partial XES trace and redirect to predict page.
 *
 * Reads config name from document.body.dataset.configName (set by petri_net.html).
 * On success: stores result in sessionStorage and redirects to /predict/<config>.
 * On error: shows the trace-error-modal with the server message.
 */

document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("partial-trace-input");
    if (!input) return;
    input.addEventListener("change", _onTraceFileSelected);
});

async function _onTraceFileSelected(e) {
    const file = e.target.files[0];
    e.target.value = "";   // reset so same file can be re-selected
    if (!file) return;

    const btn = document.getElementById("btn-predict-trace");
    if (btn) btn.classList.add("loading");

    const configName = document.body.dataset.configName;
    const formData = new FormData();
    formData.append("file", file);

    let resp;
    try {
        resp = await fetch(`/api/${configName}/replay-partial-trace`, {
            method: "POST",
            body: formData,
        });
    } catch (networkErr) {
        if (btn) btn.classList.remove("loading");
        _showTraceError("Network error: " + networkErr.message);
        return;
    }

    if (btn) btn.classList.remove("loading");

    if (!resp.ok) {
        let msg = resp.statusText;
        try { msg = (await resp.json()).error || msg; } catch (_) { /* ignore */ }
        _showTraceError(msg);
        return;
    }

    const data = await resp.json();
    sessionStorage.setItem("partialTraceInit", JSON.stringify(data));
    window.location.href = `/predict/${configName}`;
}

function _showTraceError(message) {
    const modal = document.getElementById("trace-error-modal");
    const msgEl = document.getElementById("trace-error-msg");
    if (!modal || !msgEl) { alert(message); return; }
    msgEl.textContent = message;
    modal.classList.remove("js-hidden");
}
