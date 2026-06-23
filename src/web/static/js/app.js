/**
 * Main application orchestrator.
 *
 * Handles API communication, version switching, import/export.
 * Config name is read from document.body.dataset.configName.
 */

let activeVersion = "current";
let _initPlace = null;

function apiUrl(path) {
    return `/api/${document.body.dataset.configName}${path}`;
}

async function fetchJSON(url, options) {
    const resp = await fetch(url, options);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        throw new Error(err.error || resp.statusText);
    }
    return resp.json();
}

// ---------------------------------------------------------------------------
// Place selection — drives the Predict button
// ---------------------------------------------------------------------------

function onPlaceSelected(placeId) {
    _initPlace = placeId;
    const btn = document.getElementById("btn-predict");
    if (!btn) return;
    if (activeVersion === "current") {
        btn.href = `/predict/${document.body.dataset.configName}?init_place=${encodeURIComponent(placeId)}`;
        btn.classList.remove("disabled");
    }
}

function onPlaceClear() {
    if (_initPlace === null) return;
    _initPlace = null;
    const btn = document.getElementById("btn-predict");
    if (!btn) return;
    btn.removeAttribute("href");
    btn.classList.add("disabled");
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Domain staleness banner
// ---------------------------------------------------------------------------

function setDomainStale(stale) {
    const banner = document.getElementById("domain-stale-banner");
    if (!banner) return;
    if (stale) banner.classList.add("visible");
    else banner.classList.remove("visible");
}

async function rebuildDomain() {
    const btn = document.querySelector(".btn-rebuild");
    if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Rebuilding...'; }
    try {
        const resp = await fetch(apiUrl("/rebuild-domain"), { method: "POST" });
        const data = await resp.json();
        if (data.status === "ok") {
            setDomainStale(false);
        } else if (data.error) {
            alert("Rebuild failed: " + data.error);
        }
    } catch (err) {
        alert("Rebuild failed: " + err.message);
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Rebuild Domain'; }
    }
}

// ---------------------------------------------------------------------------

async function loadVersion(version) {
    activeVersion = version;
    onPlaceClear();
    setDomainStale(false);

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

async function resetToOriginal() {
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
    const startVersion = document.body.dataset.initialVersion === "original"
        ? "original" : "current";
    loadVersion(startVersion).then(() => {
        const params = new URLSearchParams(window.location.search);
        const panel = params.get("panel");
        const panelType = params.get("type");
        if (panel && panelType === "xor") {
            showXorSplitPanel(panel);
        } else if (panel) {
            showTransitionPanel(panel);
        }
    });
});
