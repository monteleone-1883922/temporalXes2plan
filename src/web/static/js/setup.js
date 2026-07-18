/**
 * Setup page: form submission, pipeline job polling, redirect on completion.
 */

const POLL_INTERVAL_MS = 2000;
const LOG_TYPE = document.body.dataset.logType || "xes";

// For CSV logs: disable Run Analysis until required mapping fields are selected.
if (LOG_TYPE === "csv") {
    const requiredSelects = document.querySelectorAll(".csv-required-field");
    const runBtn = document.getElementById("run-btn");

    function _updateRunBtn() {
        const allFilled = Array.from(requiredSelects).every(s => s.value !== "");
        runBtn.disabled = !allFilled;
        runBtn.title = allFilled ? "" : "Complete the CSV Column Mapping above to enable analysis";
    }

    _updateRunBtn();
    requiredSelects.forEach(s => s.addEventListener("change", _updateRunBtn));
}

document.getElementById("setup-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await startPipeline();
});

// Optimizer toggle: hide the form fields the optimizer tunes when enabled,
// and show optimizer-only fields (e.g. search_n_trials) only when enabled.
const OPTIMIZER_TUNABLE_SELECTOR = '[data-optimizer-tunable="true"]';
const OPTIMIZER_ONLY_SELECTOR = '[data-optimizer-only="true"]';

function applyOptimizerMode() {
    const enabled = document.getElementById("p-use-optimizer").checked;
    document.querySelectorAll(OPTIMIZER_TUNABLE_SELECTOR).forEach((el) => {
        el.classList.toggle("js-hidden", enabled);
    });
    document.querySelectorAll(OPTIMIZER_ONLY_SELECTOR).forEach((el) => {
        el.classList.toggle("js-hidden", !enabled);
    });
}

document.getElementById("p-use-optimizer").addEventListener("change", applyOptimizerMode);
applyOptimizerMode();

async function startPipeline() {
    const form = document.getElementById("setup-form");
    const btn = document.getElementById("run-btn");
    const errorEl = document.getElementById("setup-error");
    const logName = document.body.dataset.logName;

    errorEl.classList.add("js-hidden");
    document.getElementById("setup-info").classList.add("js-hidden");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Validating...';

    // Step 1: validate log fields before starting the pipeline thread
    let validation;
    try {
        const vResp = await fetch(`/api/${logName}/validate-log`, { method: "POST" });
        validation = await vResp.json();
        if (!vResp.ok) throw new Error(validation.error || vResp.statusText);
    } catch (err) {
        showError("Validation failed: " + err.message);
        _resetRunBtn(btn);
        return;
    }

    if (validation.errors?.length) {
        showError(validation.errors.join(" — "));
        _resetRunBtn(btn);
        return;
    }

    const infoMessages = [...(validation.infos || []), ...(validation.warnings || [])];
    if (infoMessages.length) {
        document.getElementById("setup-info-text").textContent = infoMessages.join(" | ");
        document.getElementById("setup-info").classList.remove("js-hidden");
    }

    // Step 2: run pipeline
    btn.innerHTML = '<span class="spinner"></span> Starting...';
    const body = buildRequestBody(form);
    body.allow_missing_timestamp = !!validation.missing_timestamp;

    let jobId;
    try {
        const resp = await fetch(`/api/${logName}/run-pipeline`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || resp.statusText);
        jobId = data.job_id;
    } catch (err) {
        showError(err.message);
        _resetRunBtn(btn);
        return;
    }

    form.classList.add("js-hidden");
    document.getElementById("progress-panel").classList.remove("js-hidden");
    pollJob(jobId);
}

function _resetRunBtn(btn) {
    btn.disabled = LOG_TYPE === "csv"
        ? !Array.from(document.querySelectorAll(".csv-required-field")).every(s => s.value !== "")
        : false;
    btn.innerHTML = '<i class="bi bi-play-fill"></i> Run Analysis';
}


function buildRequestBody(form) {
    const useOptimizer = form["use_optimizer"].checked;

    const pipeline = {
        discovery_algorithm: form.algorithm.value,
        coverage_percentage: parseFloat(form.coverage_percentage.value),
        use_durative: form["use_durative"].checked,
        use_costs: form["use_costs"].checked,
        use_activity_classifier: form["use_activity_classifier"].checked,
        search: useOptimizer,
    };
    if (useOptimizer) {
        pipeline.search_n_trials = parseInt(form["search_n_trials"].value, 10);
    }

    const config = {};
    const numericFields = [
        "dt_min_samples", "dt_min_accuracy", "dt_max_depth",
        "probability_min_samples", "static_appearances_min_samples", "static_attr_probability",
        "kmeans_max_k", "dominance_threshold", "min_residual_points",
        "min_gvf_threshold", "gvf_target", "min_gvf_improvement",
        "kde_grid_points", "kde_extrema_order",
        "dt_prune_min_leaf_samples", "dt_prune_min_purity",
        "xor_prune_threshold", "dt_min_prob_to_use",
        "effect_never_threshold", "effect_always_threshold",
        "effect_value_prune_threshold", "effect_value_certain_threshold",
        "related_effect_prob", "incompatible_effect_prob",
        "replay_min_fitness",
        "attr_precondition_min_frequency", "attr_precondition_min_firings",
        "lower_bound_prob_actions",
    ];
    const selectFields = [
        "dt_prune_orphan_mode", "xor_statistical_mode",
        "effect_appearance_mode", "effect_value_mode",
        "replay_engine", "replay_alignment_variant",
    ];

    // Skip fields hidden by the optimizer toggle -- suggest_config() always
    // overwrites them anyway when search=True, but this keeps the payload
    // honest (only sends what's actually visible/editable on screen).
    function isHiddenByOptimizer(el) {
        if (!useOptimizer) return false;
        const wrapper = el.closest(OPTIMIZER_TUNABLE_SELECTOR);
        return wrapper !== null;
    }

    for (const name of numericFields) {
        const el = form.elements[name];
        if (el && !isHiddenByOptimizer(el)) config[name] = parseFloat(el.value);
    }
    for (const name of selectFields) {
        const el = form.elements[name];
        if (el && !isHiddenByOptimizer(el)) config[name] = el.value;
    }

    if (!isHiddenByOptimizer(form["xor_screen_prune_branches"])) {
        config["xor_screen_prune_branches"] = form["xor_screen_prune_branches"].checked;
    }
    config["log_removed_effects"] = form["log_removed_effects"].checked;
    config["snapshot_dir"] = form["snapshot_dir"].value;

    const rawAttrs = document.getElementById("c-ignored_attributes").value;
    config["ignored_attributes"] = rawAttrs
        .split("\n")
        .map(s => s.trim())
        .filter(Boolean);

    const body = { pipeline, config };

    if (LOG_TYPE === "csv") {
        body.csv_mapping = {
            case_id:   document.getElementById("csv-case_id")?.value  || null,
            activity:  document.getElementById("csv-activity")?.value  || null,
            timestamp: document.getElementById("csv-timestamp")?.value || null,
            lifecycle: document.getElementById("csv-lifecycle")?.value || null,
        };
    }

    return body;
}

function pollJob(jobId) {
    const logEl = document.getElementById("progress-log");
    const titleEl = document.getElementById("progress-title");
    let lastLogCount = 0;

    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/jobs/${jobId}`);
            const job = await resp.json();

            // Append new log lines
            const newLines = (job.log || []).slice(lastLogCount);
            for (const line of newLines) {
                const p = document.createElement("p");
                p.textContent = line;
                logEl.appendChild(p);
                logEl.scrollTop = logEl.scrollHeight;
            }
            lastLogCount = (job.log || []).length;

            if (job.status === "done") {
                clearInterval(interval);
                titleEl.textContent = "Pipeline complete!";
                titleEl.previousElementSibling.className = "bi bi-check-circle-fill";
                titleEl.previousElementSibling.style.color = "var(--green-600)";
                const configName = job.result?.config_name || document.body.dataset.logName;
                setTimeout(() => {
                    window.location.href = `/project/${configName}`;
                }, 800);
            } else if (job.status === "error") {
                clearInterval(interval);
                titleEl.textContent = "Pipeline failed";
                showError(job.error || "Unknown error");
                document.getElementById("progress-panel").classList.add("js-hidden");
                document.getElementById("setup-form").classList.remove("js-hidden");
                const btn = document.getElementById("run-btn");
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-play-fill"></i> Run Analysis';
            }
        } catch (_) {
            // network glitch — keep polling
        }
    }, POLL_INTERVAL_MS);
}

function showError(msg) {
    const el = document.getElementById("setup-error");
    el.textContent = msg;
    el.classList.remove("js-hidden");
}
