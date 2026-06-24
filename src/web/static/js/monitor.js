/**
 * Predictive monitoring panel: loads Petri net data, builds init/goal forms,
 * submits to /build-problem, and displays the generated problem.pddl.
 *
 * Init  — optional starting place (marked) + attribute value assignments.
 * Goal  — SOP conditions: [[{attribute, predicate:"="/"<>", value}], ...]
 */

let _petriData = null;
let _catalog = {};
let _initPlace = null;
let _plannersInfo = null;
let _plannerConfig = null;
let _goalBuilder = null;
let _initBuilder = null;

document.addEventListener("DOMContentLoaded", async () => {
    _initPlace = new URLSearchParams(window.location.search).get("init_place");

    if (_initPlace) {
        const banner = document.getElementById("init-place-banner");
        document.getElementById("init-place-label").textContent = _initPlace;
        banner.classList.remove("js-hidden");
    }

    try {
        await Promise.all([loadPlannersInfo(), loadPlannerConfig(), loadPetriData()]);
        buildMonitorForms();
        document.getElementById("monitor-loading").classList.add("js-hidden");
        document.getElementById("monitor-forms").classList.remove("js-hidden");
    } catch (err) {
        document.getElementById("monitor-loading").textContent =
            "Failed to load data: " + err.message;
    }

    document.getElementById("solve-form").addEventListener("submit", onSolveSubmit);
});

async function loadPetriData() {
    const resp = await fetch(`/api/${document.body.dataset.configName}/petri-net`);
    if (!resp.ok) throw new Error("Could not load Petri net data");
    _petriData = await resp.json();
    _catalog = _petriData.attribute_catalog || {};
}

// ---------------------------------------------------------------------------
// Form builders
// ---------------------------------------------------------------------------

function buildMonitorForms() {
    buildInitForm();
    buildGoalSopForm();
    buildPlannerSection();
}

// ---------------------------------------------------------------------------
// Init form — flat attribute=value list
// ---------------------------------------------------------------------------

function buildInitForm() {
    const container = document.getElementById("init-predicates");
    container.innerHTML = "";

    if (Object.keys(_catalog).length === 0) {
        container.innerHTML = '<p class="field-hint">No attributes in this configuration.</p>';
        return;
    }

    _initBuilder = createAssignmentList(_catalog, []);
    container.appendChild(_initBuilder.el);
}

// ---------------------------------------------------------------------------
// Goal form — SOP builder
// ---------------------------------------------------------------------------

function buildGoalSopForm() {
    const container = document.getElementById("goal-predicates");
    container.innerHTML = "";
    _goalBuilder = createSopBuilder(_catalog, [], { withPredicate: true });
    container.appendChild(_goalBuilder.el);
}

// ---------------------------------------------------------------------------
// Data collection
// ---------------------------------------------------------------------------

function collectInitEffects() {
    return _initBuilder ? _initBuilder.getValue() : [];
}

function collectGoalSop() {
    return _goalBuilder ? _goalBuilder.getValue() : [];
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

async function onSolveSubmit(e) {
    e.preventDefault();

    const btn = document.getElementById("solve-btn");
    const errorEl = document.getElementById("solve-error");
    const resultEl = document.getElementById("solve-result");
    const runBtn = document.getElementById("run-planner-btn");

    errorEl.classList.add("js-hidden");
    resultEl.classList.add("js-hidden");
    if (runBtn) runBtn.disabled = true;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Building...';

    const init = collectInitEffects();
    const goal = collectGoalSop();
    const metric = (document.getElementById("metric-select")?.value) || null;
    const requireCompletion = document.getElementById("require-completion-checkbox")?.checked ?? false;

    if (goal.length === 0 && !requireCompletion) {
        showSolveError("Add at least one goal clause or enable require completion.");
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-hammer"></i> Build Problem';
        return;
    }

    try {
        const resp = await fetch(`/api/${document.body.dataset.configName}/build-problem`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ init_place: _initPlace, init, goal, metric, require_completion: requireCompletion }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || resp.statusText);
        showSolveResult(data);
    } catch (err) {
        showSolveError(err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-hammer"></i> Build Problem';
    }
}

function showSolveError(msg) {
    const el = document.getElementById("solve-error");
    el.textContent = msg;
    el.classList.remove("js-hidden");
}

function showSolveResult(data) {
    const el = document.getElementById("solve-result");
    el.classList.remove("js-hidden");
    el.innerHTML = `
        <div class="result-status result-built">
            <i class="bi bi-file-earmark-check-fill"></i>
            problem.pddl written successfully
        </div>
        <details class="pddl-preview">
            <summary>Show generated problem.pddl</summary>
            <pre>${escapeHtml(data.problem_text)}</pre>
        </details>`;

    const runBtn = document.getElementById("run-planner-btn");
    if (runBtn) runBtn.disabled = false;
}

function escapeHtml(s) {
    return s
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

// =============================================================================
// Planner section
// =============================================================================

async function loadPlannersInfo() {
    try {
        const resp = await fetch("/api/planners");
        if (resp.ok) _plannersInfo = await resp.json();
    } catch (_) { /* non-fatal */ }
}

async function loadPlannerConfig() {
    try {
        const resp = await fetch(`/api/${document.body.dataset.configName}/planner-config`);
        if (resp.ok) _plannerConfig = await resp.json();
    } catch (_) { /* non-fatal — defaults used */ }
}

function buildPlannerSection() {
    const container = document.getElementById("planner-inner");
    container.innerHTML = "";

    const fd = (_plannersInfo || {}).fast_downward || {};
    const op = (_plannersInfo || {}).optic || {};
    const fdCfg = (_plannerConfig || {}).fast_downward || {};
    const opCfg = (_plannerConfig || {}).optic || {};

    // ── Planner tabs ──────────────────────────────────────────────────────────
    const tabs = document.createElement("div");
    tabs.className = "planner-tabs";

    const tabFd = _makeTab("Fast Downward", "fast_downward");
    const tabOp = _makeTab("OPTIC", "optic");
    tabs.appendChild(tabFd);
    tabs.appendChild(tabOp);
    container.appendChild(tabs);

    // ── Options panels ────────────────────────────────────────────────────────
    const panelFd = _buildFdPanel(fd, fdCfg);
    panelFd.id = "planner-opts-fast_downward";
    container.appendChild(panelFd);
    _populateFdFields(fdCfg);

    const panelOp = _buildOpticPanel(op, opCfg);
    panelOp.id = "planner-opts-optic";
    panelOp.style.display = "none";
    container.appendChild(panelOp);
    _populateOpticFields(opCfg);

    // Tab switching
    [tabFd, tabOp].forEach(tab => {
        tab.addEventListener("click", () => {
            [tabFd, tabOp].forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            panelFd.style.display = tab.dataset.planner === "fast_downward" ? "block" : "none";
            panelOp.style.display = tab.dataset.planner === "optic"          ? "block" : "none";
            _updateMetricOptions(tab.dataset.planner);
        });
    });

    tabFd.click();

    // Wire the static run button (replace to avoid duplicate listeners on re-build)
    const oldBtn = document.getElementById("run-planner-btn");
    if (oldBtn) {
        const newBtn = oldBtn.cloneNode(true);
        oldBtn.replaceWith(newBtn);
        newBtn.addEventListener("click", onPlannerRun);
    }
}

function _updateMetricOptions(planner) {
    const sel = document.getElementById("metric-select");
    if (!sel) return;
    const timeOpt = sel.querySelector('option[value="minimize_time"]');
    if (!timeOpt) return;
    const isFd = planner === "fast_downward";
    timeOpt.disabled = isFd;
    if (isFd && sel.value === "minimize_time") sel.value = "";
}

function _makeTab(label, planner) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "planner-tab";
    btn.dataset.planner = planner;
    btn.textContent = label;
    return btn;
}

function _buildFdPanel(fd, fdCfg) {
    const panel = document.createElement("div");
    panel.className = "planner-options-panel";

    const configs = fd.search_configs || {};
    const groups = {};
    for (const [key, cfg] of Object.entries(configs)) {
        (groups[cfg.group] = groups[cfg.group] || []).push({ key, ...cfg });
    }

    // Panel header with reset button
    const header = document.createElement("div");
    header.className = "planner-panel-header";
    header.innerHTML = `<span class="planner-panel-title">Fast Downward options</span>`;
    const resetBtn = _makeResetBtn("fast_downward");
    header.appendChild(resetBtn);
    panel.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "planner-opts-grid";

    // Search algorithm select
    const fieldAlg = document.createElement("div");
    fieldAlg.className = "form-field";
    fieldAlg.innerHTML = "<label>Search algorithm</label>";
    const sel = document.createElement("select");
    sel.id = "fd-search-select";
    sel.className = "form-control";
    for (const [group, entries] of Object.entries(groups)) {
        const og = document.createElement("optgroup");
        og.label = group;
        for (const e of entries) {
            const opt = document.createElement("option");
            opt.value = e.key;
            opt.textContent = e.key;
            og.appendChild(opt);
        }
        sel.appendChild(og);
    }
    fieldAlg.appendChild(sel);
    grid.appendChild(fieldAlg);

    // Timeout
    const fieldTo = document.createElement("div");
    fieldTo.className = "form-field";
    fieldTo.innerHTML = `<label>Timeout (s)</label>
        <input id="fd-timeout" type="number" class="form-control" min="1" max="10800">`;
    grid.appendChild(fieldTo);

    // Memory limit (optional)
    const fieldMem = document.createElement("div");
    fieldMem.className = "form-field";
    fieldMem.innerHTML = `<label>Memory limit (MB) <span class="field-hint-inline">optional</span></label>
        <input id="fd-memory" type="number" class="form-control" min="256" placeholder="no limit">`;
    grid.appendChild(fieldMem);

    panel.appendChild(grid);
    return panel;
}

function _buildOpticPanel(op, opCfg) {
    const panel = document.createElement("div");
    panel.className = "planner-options-panel";

    // Panel header with reset button
    const header = document.createElement("div");
    header.className = "planner-panel-header";
    header.innerHTML = `<span class="planner-panel-title">OPTIC options</span>`;
    const resetBtn = _makeResetBtn("optic");
    header.appendChild(resetBtn);
    panel.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "planner-opts-grid";
    grid.innerHTML = `
        <div class="form-field">
            <label>Timeout (s)</label>
            <input id="optic-timeout" type="number" class="form-control" min="1" max="10800">
        </div>
        <div class="form-field">
            <label>Memory limit (MB)</label>
            <input id="optic-memory" type="number" class="form-control" min="256">
        </div>`;
    panel.appendChild(grid);

    const checks = document.createElement("div");
    checks.className = "planner-checks";
    checks.innerHTML = `
        <label class="check-label">
            <input type="checkbox" id="optic-stop-first">
            Stop at first solution <span class="field-hint-inline">(-N, skip cost optimisation)</span>
        </label>
        <label class="check-label">
            <input type="checkbox" id="optic-ignore-costs">
            Ignore action costs <span class="field-hint-inline">(-c, treat all actions as unit cost)</span>
        </label>`;
    panel.appendChild(checks);
    return panel;
}

// ---------------------------------------------------------------------------
// Field populate helpers (called at build time and after reset)
// ---------------------------------------------------------------------------

function _populateFdFields(cfg) {
    const sel = document.getElementById("fd-search-select");
    if (sel) sel.value = cfg.search || "astar_lmcut";
    const to = document.getElementById("fd-timeout");
    if (to) to.value = cfg.timeout ?? 30;
    const mem = document.getElementById("fd-memory");
    if (mem) mem.value = cfg.memory_mb != null ? cfg.memory_mb : "";
}

function _populateOpticFields(cfg) {
    const to = document.getElementById("optic-timeout");
    if (to) to.value = cfg.timeout ?? 60;
    const mem = document.getElementById("optic-memory");
    if (mem) mem.value = cfg.memory_mb ?? 4000;
    const stop = document.getElementById("optic-stop-first");
    if (stop) stop.checked = cfg.stop_at_first ?? true;
    const ic = document.getElementById("optic-ignore-costs");
    if (ic) ic.checked = cfg.ignore_costs ?? false;
}

// ---------------------------------------------------------------------------
// Reset button
// ---------------------------------------------------------------------------

function _makeResetBtn(planner) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-reset-config";
    btn.innerHTML = '<i class="bi bi-arrow-counterclockwise"></i> Reset to defaults';
    btn.addEventListener("click", async () => {
        try {
            const resp = await fetch(`/api/${document.body.dataset.configName}/planner-config/reset`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ planner }),
            });
            if (!resp.ok) return;
            const data = await resp.json();
            _plannerConfig = data;
            if (planner === "fast_downward") _populateFdFields(data.fast_downward || {});
            else _populateOpticFields(data.optic || {});
        } catch (err) {
            console.error("Config reset failed:", err);
        }
    });
    return btn;
}

function _activePlanner() {
    const active = document.querySelector(".planner-tab.active");
    return active ? active.dataset.planner : "fast_downward";
}

async function onPlannerRun() {
    const planner = _activePlanner();

    const btn = document.getElementById("run-planner-btn");
    const errEl = document.getElementById("planner-error");
    const progress = document.getElementById("planner-progress");
    const result = document.getElementById("planner-result");

    errEl.style.display = "none";
    result.style.display = "none";
    progress.style.display = "block";
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running...';
    document.getElementById("planner-log").innerHTML = "";
    document.getElementById("planner-progress-title").textContent = "Running planner...";

    const options = _collectPlannerOptions(planner);

    // Persist the current options before launching the job.
    try {
        const updatedConfig = {
            fast_downward: { ...(_plannerConfig?.fast_downward || {}), ...(planner === "fast_downward" ? options : {}) },
            optic: { ...(_plannerConfig?.optic || {}), ...(planner === "optic" ? options : {}) },
        };
        const saveResp = await fetch(`/api/${document.body.dataset.configName}/planner-config`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(updatedConfig),
        });
        if (saveResp.ok) _plannerConfig = await saveResp.json();
    } catch (_) { /* non-fatal — run anyway */ }

    try {
        const resp = await fetch(`/api/${document.body.dataset.configName}/run-planner`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ planner, options }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || resp.statusText);
        _pollPlannerJob(data.job_id);
    } catch (err) {
        _showPlannerError(err.message);
        progress.style.display = "none";
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-fill"></i> Run Planner';
    }
}

function _collectPlannerOptions(planner) {
    if (planner === "fast_downward") {
        const sel = document.getElementById("fd-search-select");
        const to = document.getElementById("fd-timeout");
        const mem = document.getElementById("fd-memory");
        const memStr = mem ? mem.value.trim() : "";
        return {
            search: sel ? sel.value : "astar_lmcut",
            timeout: to ? parseInt(to.value) || 30 : 30,
            memory_mb: memStr !== "" ? parseInt(memStr) : null,
        };
    }
    if (planner === "optic") {
        return {
            stop_at_first: document.getElementById("optic-stop-first")?.checked ?? true,
            ignore_costs: document.getElementById("optic-ignore-costs")?.checked ?? false,
            timeout: parseInt(document.getElementById("optic-timeout")?.value || "60") || 60,
            memory_mb: parseInt(document.getElementById("optic-memory")?.value || "4000") || 4000,
        };
    }
    return {};
}

function _pollPlannerJob(jobId) {
    const logEl = document.getElementById("planner-log");
    const titleEl = document.getElementById("planner-progress-title");
    const iconEl = document.getElementById("planner-progress-icon");
    let lastCount = 0;
    const INTERVAL = 2000;

    const iv = setInterval(async () => {
        try {
            const resp = await fetch(`/api/jobs/${jobId}`);
            const job = await resp.json();

            const newLines = (job.log || []).slice(lastCount);
            for (const line of newLines) {
                const p = document.createElement("p");
                p.textContent = line;
                logEl.appendChild(p);
                logEl.scrollTop = logEl.scrollHeight;
            }
            lastCount = (job.log || []).length;

            if (job.status === "done") {
                clearInterval(iv);
                titleEl.textContent = job.result?.success ? "Plan found!" : "Planning complete";
                iconEl.className = job.result?.success
                    ? "bi bi-check-circle-fill"
                    : "bi bi-x-circle-fill";
                iconEl.style.color = job.result?.success
                    ? "var(--green-600)" : "var(--red-600)";
                document.getElementById("run-planner-btn").disabled = false;
                document.getElementById("run-planner-btn").innerHTML =
                    '<i class="bi bi-play-fill"></i> Run Planner';
                _showPlanResult(job.result);
            } else if (job.status === "error") {
                clearInterval(iv);
                _showPlannerError(job.error || "Unknown error");
                document.getElementById("planner-progress").style.display = "none";
                document.getElementById("run-planner-btn").disabled = false;
                document.getElementById("run-planner-btn").innerHTML =
                    '<i class="bi bi-play-fill"></i> Run Planner';
            }
        } catch (_) { /* network glitch — keep polling */ }
    }, INTERVAL);
}

function _showPlannerError(msg) {
    const el = document.getElementById("planner-error");
    el.textContent = msg;
    el.style.display = "block";
}

function _showPlanResult(result) {
    const el = document.getElementById("planner-result");
    el.style.display = "block";

    if (!result || !result.success) {
        const label = result?.solvability === "unsolvable_structural"
            ? "The problem is provably unsolvable with the current init and goal."
            : "No plan found within the given resource limits.";
        el.innerHTML = `
            <div class="result-status" style="background:var(--red-100);color:var(--red-600);border-color:var(--red-100);">
                <i class="bi bi-x-circle-fill"></i> ${label}
            </div>`;
        return;
    }

    const actions = result.plan_actions || [];
    const metrics = result.metrics || {};

    const metaParts = [];
    if (metrics.solution_length != null) metaParts.push(`${metrics.solution_length} step(s)`);
    if (metrics.total_time != null)      metaParts.push(`${metrics.total_time.toFixed(2)}s`);
    if (metrics.expanded_nodes != null)  metaParts.push(`${metrics.expanded_nodes} nodes`);

    const actionItems = actions.map((a, i) =>
        `<li class="plan-action-item"><span class="plan-step">${i + 1}</span>${escapeHtml(a)}</li>`
    ).join("");

    el.innerHTML = `
        <div class="result-status result-built">
            <i class="bi bi-check-circle-fill"></i> Plan found
            ${metaParts.length ? `<span class="plan-meta">${metaParts.join(" · ")}</span>` : ""}
        </div>
        ${actions.length
            ? `<ol class="plan-action-list">${actionItems}</ol>`
            : `<p class="field-hint">No visible actions (all steps are silent transitions).</p>`
        }
        ${result.plan_text
            ? `<details class="pddl-preview" style="margin-top:12px;">
                   <summary>Show raw plan</summary>
                   <pre>${escapeHtml(result.plan_text)}</pre>
               </details>`
            : ""
        }`;
}
