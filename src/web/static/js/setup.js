/**
 * Setup page: form submission, pipeline job polling, redirect on completion.
 */

const POLL_INTERVAL_MS = 2000;

document.getElementById("setup-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await startPipeline();
});

async function startPipeline() {
    const form = document.getElementById("setup-form");
    const btn = document.getElementById("run-btn");
    const errorEl = document.getElementById("setup-error");

    errorEl.classList.add("js-hidden");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Starting...';

    const body = buildRequestBody(form);

    let jobId;
    try {
        const resp = await fetch(`/api/${document.body.dataset.logName}/run-pipeline`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || resp.statusText);
        jobId = data.job_id;
    } catch (err) {
        showError(err.message);
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-fill"></i> Run Analysis';
        return;
    }

    form.classList.add("js-hidden");
    document.getElementById("progress-panel").classList.remove("js-hidden");
    pollJob(jobId);
}

function buildRequestBody(form) {
    const pipeline = {
        discovery_algorithm: form.algorithm.value,
        coverage_percentage: parseFloat(form.coverage_percentage.value),
        use_durative: form["use_durative"].checked,
        use_activity_classifier: form["use_activity_classifier"].checked,
    };

    const config = {};
    const numericFields = [
        "dt_min_samples", "dt_min_accuracy", "dt_max_depth",
        "probability_min_samples",
        "kmeans_max_k", "kmeans_min_cluster_fraction",
        "kmeans_silhouette_threshold", "kmeans_n_init",
        "dt_prune_min_leaf_samples", "dt_prune_min_purity",
        "xor_prune_threshold",
        "effect_never_threshold", "effect_always_threshold",
        "effect_appearance_threshold",
        "effect_value_prune_threshold", "effect_value_certain_threshold",
        "related_effect_prob", "incompatible_effect_prob",
        "replay_min_fitness",
        "attr_precondition_min_frequency", "attr_precondition_min_firings",
    ];
    const selectFields = [
        "dt_prune_orphan_mode", "xor_statistical_mode",
        "effect_appearance_mode", "effect_value_mode",
    ];

    for (const name of numericFields) {
        const el = form.elements[name];
        if (el) config[name] = parseFloat(el.value);
    }
    for (const name of selectFields) {
        const el = form.elements[name];
        if (el) config[name] = el.value;
    }

    config["log_removed_effects"] = form["log_removed_effects"].checked;
    config["snapshot_dir"] = form["snapshot_dir"].value;

    const rawAttrs = document.getElementById("c-ignored_attributes").value;
    config["ignored_attributes"] = rawAttrs
        .split("\n")
        .map(s => s.trim())
        .filter(Boolean);

    return { pipeline, config };
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
