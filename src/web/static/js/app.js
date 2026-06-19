/**
 * Main application orchestrator.
 *
 * Handles API communication, version switching, import/export.
 * CONFIG_NAME is set inline by the template before this script loads.
 */

let activeVersion = "current";

function apiUrl(path) {
    return `/api/${CONFIG_NAME}${path}`;
}

async function fetchJSON(url, options) {
    const resp = await fetch(url, options);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        throw new Error(err.error || resp.statusText);
    }
    return resp.json();
}

async function loadVersion(version) {
    activeVersion = version;

    const endpoint =
        version === "original"
            ? apiUrl("/petri-net/original")
            : apiUrl("/petri-net");

    try {
        const data = await fetchJSON(endpoint);
        setCurrentData(data);
        renderGraph(data);
        closePanel();
        updateToolbarButtons();
    } catch (err) {
        alert("Failed to load data: " + err.message);
    }
}

function updateToolbarButtons() {
    const btnCurrent = document.getElementById("btn-current");
    const btnOriginal = document.getElementById("btn-original");
    if (activeVersion === "current") {
        btnCurrent.classList.add("active");
        btnOriginal.classList.remove("active");
    } else {
        btnCurrent.classList.remove("active");
        btnOriginal.classList.add("active");
    }
}

async function resetToCurrent() {
    if (!confirm("Reset all changes to the original version?")) return;
    try {
        await fetchJSON(apiUrl("/petri-net/reset"), { method: "POST" });
        await loadVersion("current");
    } catch (err) {
        alert("Reset failed: " + err.message);
    }
}

function exportConfig() {
    window.location.href = apiUrl("/export");
}

async function importConfig(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        await fetchJSON(apiUrl("/import"), { method: "POST", body: formData });
        await loadVersion("current");
    } catch (err) {
        alert("Import failed: " + err.message);
    }

    event.target.value = "";
}

// Boot
document.addEventListener("DOMContentLoaded", () => {
    initGraph();
    const startVersion = (typeof INITIAL_VERSION !== "undefined" && INITIAL_VERSION === "original")
        ? "original" : "current";
    loadVersion(startVersion).then(() => {
        const params = new URLSearchParams(window.location.search);
        const panel = params.get("panel");
        const panelType = params.get("type");
        if (panel && panelType === "xor") {
            showXorSplitPanel(panel);
        } else if (panel && panelType === "art") {
            showArtificialXorPanel(panel);
        } else if (panel) {
            showTransitionPanel(panel);
        }
    });
});
