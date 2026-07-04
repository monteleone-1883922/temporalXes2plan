/**
 * Detail panels for transitions and XOR splits — styled version.
 */

let currentData = null;
let _panelEntity = null;

function setCurrentData(data) { currentData = data; }

function closePanel() {
    document.getElementById("detail-panel").classList.add("d-none");
    document.getElementById("panel-content").innerHTML = "";
    _panelEntity = null;
    _showEditBtn(false);
    if (cy) cy.nodes(":selected").unselect();
}

function _showEditBtn(show) {
    const btn = document.getElementById("panel-edit-btn");
    if (btn) btn.style.display = show ? "inline-flex" : "none";
}

function enterEditMode() {
    if (!_panelEntity) return;
    if (_panelEntity.type === "transition") _enterTransitionEdit(_panelEntity.id);
    else if (_panelEntity.type === "xor") _enterXorEdit(_panelEntity.id);
}

// ── Section icons ──────────────────────────────────────────────────────────
const SECTION_ICONS = {
    "Analysis Info":      "bi-bar-chart-line",
    "Duration":           "bi-clock",
    "Preconditions":      "bi-funnel",
    "Effects":            "bi-lightning-charge",
    "Cost":               "bi-tag",
    "Guards":             "bi-signpost-split",
    "Variant Conditions": "bi-funnel",
    "Variant Effects":    "bi-lightning-charge",
};

// ─────────────────────────────────────────────────────────────────────────────
// Transition panel
// ─────────────────────────────────────────────────────────────────────────────

function showTransitionPanel(activityName) {
    if (!currentData) return;
    const t = currentData.transitions[activityName];
    if (!t) return;

    const icon = document.getElementById("panel-header-icon");
    icon.className = "panel-header-icon";
    icon.innerHTML = '<i class="bi bi-toggles"></i>';

    document.getElementById("panel-title").textContent = t.activity_name;
    const content = document.getElementById("panel-content");
    content.innerHTML = "";

    content.appendChild(buildMetaSection(t));
    content.appendChild(buildDurationSection(t.duration));
    content.appendChild(buildPreconditionsSection(t.preconditions));
    content.appendChild(buildEffectsSection(t.effect_groups));
    content.appendChild(buildCostSection(t.cost));

    _panelEntity = { type: "transition", id: activityName };
    _showEditBtn(true);
    document.getElementById("detail-panel").classList.remove("d-none");
}

function buildMetaSection(t) {
    const section = createSection("Analysis Info", true);
    const grid = document.createElement("div");
    grid.className = "meta-block";

    const rows = [
        ["Firings", t._meta.total_firings],
    ];

    for (const [k, v] of rows) {
        const key = document.createElement("span");
        key.className = "meta-key";
        key.textContent = k;

        const val = document.createElement("span");
        val.className = "meta-val";
        if (typeof v === "string" && v.startsWith("<")) {
            val.innerHTML = v;
        } else {
            val.textContent = v;
        }

        grid.appendChild(key);
        grid.appendChild(val);
    }

    section.appendChild(grid);
    return section;
}

function buildDurationSection(dur) {
    const section = createSection("Duration", true);

    if (!dur) {
        section.appendChild(emptyNote("No duration data — enter values manually via Edit"));
        return section;
    }

    const grid = document.createElement("div");
    grid.className = "meta-block";

    const rows = [
        ["Min", `${dur.effective_min}s`],
        ["Max", `${dur.effective_max}s`],
        ["Source", dur.source],
    ];
    if (dur.mean !== undefined) {
        rows.push(["Mean", `${dur.mean}s`], ["Std dev", `${dur.std_dev}s`]);
    }

    for (const [k, v] of rows) {
        const key = document.createElement("span");
        key.className = "meta-key";
        key.textContent = k;
        const val = document.createElement("span");
        val.className = "meta-val";
        val.textContent = v;
        grid.appendChild(key);
        grid.appendChild(val);
    }

    section.appendChild(grid);
    return section;
}

function buildPreconditionsSection(preconditions) {
    const section = createSection("Preconditions");

    if (!preconditions || preconditions.length === 0) {
        section.appendChild(emptyNote("No preconditions"));
        return section;
    }

    renderConditionGroups(section, preconditions);
    return section;
}

function buildEffectsSection(effectGroups) {
    const section = createSection("Effects");


    if (effectGroups.length === 0) {
        section.appendChild(emptyNote("No effects"));
        return section;
    }

    _renderEffectGroups(section, effectGroups);
    return section;
}

function _renderEffectGroups(container, effectGroups) {
    for (let i = 0; i < effectGroups.length; i++) {
        if (i > 0) container.appendChild(orSeparator());
        const group = effectGroups[i];

        const groupDiv = document.createElement("div");
        groupDiv.className = "condition-group";

        for (const { attribute, value } of (group.assignments || [])) {
            groupDiv.appendChild(_effectAssignmentPill(attribute, value));
        }

        if (group.probability != null) {
            const badge = document.createElement("span");
            badge.className = "prob-badge";
            badge.style.cssText = "background:#6366f1;font-size:0.7rem;";
            badge.textContent = `${(group.probability * 100).toFixed(1)}%`;
            groupDiv.appendChild(badge);
        }

        container.appendChild(groupDiv);

        if (group.guard?.length > 0) {
            const whenLabel = document.createElement("div");
            whenLabel.className = "cond-label";
            whenLabel.style.cssText = "font-size:0.7rem;margin-top:2px;margin-left:4px;";
            whenLabel.textContent = "when:";
            container.appendChild(whenLabel);
            renderConditionGroups(container, group.guard);
        }
    }
}

function _effectAssignmentPill(attribute, value) {
    const span = document.createElement("span");
    span.className = "condition-pill condition-pill-eq";
    span.textContent = `${attribute} → ${value}`;
    return span;
}

function buildCostSection(cost) {
    const section = createSection("Cost");
    const val = document.createElement("div");
    val.className = `cost-value${cost === 0 ? " zero" : ""}`;
    val.textContent = cost.toFixed(2);
    section.appendChild(val);
    return section;
}

// ─────────────────────────────────────────────────────────────────────────────
// XOR split panel
// ─────────────────────────────────────────────────────────────────────────────

const _ROUTING_SOURCE_LABEL = {
    "dt":                  null,
    "dt_orphan":           "Fallback — DT orphan (all leaves pruned)",
    "dt_low_prob":         "Fallback — low probability (excluded from DT)",
    "deterministic":       "Deterministic (certain branch)",
    "deterministic_floor": "Fallback — floor cost (non-certain branch)",
    "probabilistic":       "Probabilistic fallback",
    "equal_weight":        "Fallback — equal weight (insufficient samples)",
};

function showXorSplitPanel(placeId) {
    if (!currentData) return;
    const xor = currentData.xor_splits[placeId];
    if (!xor) return;

    const icon = document.getElementById("panel-header-icon");
    icon.className = "panel-header-icon xor";
    icon.innerHTML = '<i class="bi bi-signpost-split"></i>';

    document.getElementById("panel-title").textContent = `XOR: ${placeId}`;
    const content = document.getElementById("panel-content");
    content.innerHTML = "";

    for (const branch of Object.values(xor.branches)) {
        const card = document.createElement("div");
        card.className = "branch-card";

        const headerRow = document.createElement("div");
        headerRow.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;";
        const name = document.createElement("span");
        name.className = "branch-name";
        name.textContent = branch.activity_name;
        const badge = document.createElement("span");
        badge.className = "prob-badge";
        badge.style.background = "#f59e0b";
        badge.textContent = `${(branch.probability * 100).toFixed(1)}%`;
        headerRow.appendChild(name);
        headerRow.appendChild(badge);
        card.appendChild(headerRow);

        if (branch.conditions && branch.conditions.length > 0) {
            const label = document.createElement("div");
            label.className = "cond-label";
            label.textContent = "Guards";
            card.appendChild(label);
            renderConditionGroups(card, branch.conditions);
        } else {
            const src = branch._meta && branch._meta.routing_source;
            const noteText = (src && src in _ROUTING_SOURCE_LABEL && _ROUTING_SOURCE_LABEL[src] !== null)
                ? _ROUTING_SOURCE_LABEL[src]
                : "No guards — statistical fallback";
            card.appendChild(emptyNote(noteText));
        }

        const meta = document.createElement("div");
        meta.className = "branch-meta";
        meta.innerHTML = `
            <span class="meta-key">Cascade</span> ${cascadeBadge(branch._meta.cascade_level)}
            <span style="color:#cbd5e1">·</span>
            <span class="meta-key">Samples</span>
            <span>${branch._meta.total_samples}</span>
        `;
        card.appendChild(meta);

        content.appendChild(card);
    }

    _panelEntity = { type: "xor", id: placeId };
    _showEditBtn(true);
    document.getElementById("detail-panel").classList.remove("d-none");
}

// ─────────────────────────────────────────────────────────────────────────────
// Edit mode
// ─────────────────────────────────────────────────────────────────────────────

function _enterTransitionEdit(actName) {
    const t = currentData.transitions[actName];
    if (!t) return;
    const catalog = currentData.attribute_catalog || {};
    const content = document.getElementById("panel-content");
    content.innerHTML = "";

    // Cost
    const costSection = _makeEditSection("Cost", "bi-tag");
    const costInput = document.createElement("input");
    costInput.type = "number";
    costInput.className = "form-control prob-input";
    costInput.min = "0";
    costInput.step = "0.01";
    costInput.value = t.cost ?? 0;
    costInput.style.maxWidth = "120px";
    costSection.appendChild(costInput);
    content.appendChild(costSection);

    // Duration
    const durSection = _makeEditSection("Duration", "bi-clock");
    const durGrid = document.createElement("div");
    durGrid.style.cssText = "display:grid;grid-template-columns:80px 1fr;align-items:center;gap:6px 10px;";

    function _durLabel(text) {
        const l = document.createElement("span");
        l.className = "meta-key";
        l.textContent = text;
        return l;
    }
    function _durInput(val) {
        const inp = document.createElement("input");
        inp.type = "number";
        inp.className = "prob-input";
        inp.min = "0";
        inp.step = "0.1";
        inp.style.maxWidth = "120px";
        if (val !== undefined && val !== null) inp.value = val;
        inp.placeholder = "seconds";
        return inp;
    }

    const durMinInput = _durInput(t.duration?.effective_min);
    const durMaxInput = _durInput(t.duration?.effective_max);

    durGrid.appendChild(_durLabel("Min (s)"));
    durGrid.appendChild(durMinInput);
    durGrid.appendChild(_durLabel("Max (s)"));
    durGrid.appendChild(durMaxInput);

    if (t.duration?.source && t.duration.source !== "external") {
        const sourceRow = document.createElement("span");
        sourceRow.className = "meta-key";
        sourceRow.textContent = "Source";
        const sourceVal = document.createElement("span");
        sourceVal.style.cssText = "font-size:0.75rem;color:var(--slate-500);";
        sourceVal.textContent = `${t.duration.source} (read-only stats: mean ${t.duration.mean}s, σ ${t.duration.std_dev}s)`;
        durGrid.appendChild(sourceRow);
        durGrid.appendChild(sourceVal);
    }

    durSection.appendChild(durGrid);
    content.appendChild(durSection);

    // Preconditions
    const precSection = _makeEditSection("Preconditions", "bi-funnel");
    const sopBuilder = createSopBuilder(catalog, t.preconditions || [], {
        withPredicate: true,
        startEmpty: !t.preconditions?.length,
    });
    precSection.appendChild(sopBuilder.el);
    content.appendChild(precSection);

    // Effects
    const effSection = _makeEditSection("Effects", "bi-lightning-charge");
    const effEditor = createEffectGroupEditor(catalog, t.effect_groups || [], t.effects || {});
    effSection.appendChild(effEditor.el);
    content.appendChild(effSection);

    content.appendChild(_buildEditActions(
        async () => {
            const body = {
                cost: parseFloat(costInput.value) || 0,
                preconditions: sopBuilder.getValue(),
                effect_groups: effEditor.getValue(),
            };

            const dMin = durMinInput.value.trim();
            const dMax = durMaxInput.value.trim();
            if (dMin !== "" && dMax !== "") {
                const effMin = parseFloat(dMin);
                const effMax = parseFloat(dMax);
                if (isNaN(effMin) || isNaN(effMax) || effMin < 0 || effMax < effMin) {
                    throw new Error("Duration: min must be ≥ 0 and max must be ≥ min");
                }
                body.duration = { effective_min: effMin, effective_max: effMax };
            }

            const resp = await fetch(`/api/${document.body.dataset.configName}/transition/${encodeURIComponent(actName)}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.error || resp.statusText);
            }
            Object.assign(currentData.transitions[actName], body);
            if (body.duration) {
                currentData.transitions[actName].duration = {
                    ...body.duration,
                    source: "external",
                };
            }
            if (typeof setDomainStale === "function") setDomainStale(true);
            showTransitionPanel(actName);
        },
        () => showTransitionPanel(actName)
    ));
}

function _enterXorEdit(placeId) {
    const xor = currentData.xor_splits[placeId];
    if (!xor) return;
    const catalog = currentData.attribute_catalog || {};
    const content = document.getElementById("panel-content");
    content.innerHTML = "";

    const branchEditors = [];

    for (const [branchActivity, branch] of Object.entries(xor.branches)) {
        const section = _makeEditSection(branch.activity_name, "bi-signpost-split");

        // Probability row
        const probWrap = document.createElement("div");
        probWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:10px;";
        const probLabel = document.createElement("span");
        probLabel.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--slate-600);min-width:80px;";
        probLabel.textContent = "Probability";
        const probInput = document.createElement("input");
        probInput.type = "number";
        probInput.className = "prob-input";
        probInput.min = "0";
        probInput.max = "1";
        probInput.step = "0.0001";
        probInput.value = branch.probability;
        probWrap.appendChild(probLabel);
        probWrap.appendChild(probInput);
        section.appendChild(probWrap);

        // Guards SOP builder
        const condLabel = document.createElement("div");
        condLabel.className = "cond-label";
        condLabel.textContent = "Guards";
        section.appendChild(condLabel);
        const sopBuilder = createSopBuilder(catalog, branch.conditions || [], { withPredicate: true });
        section.appendChild(sopBuilder.el);

        content.appendChild(section);
        branchEditors.push({ branchActivity, probInput, sopBuilder });
    }

    content.appendChild(_buildEditActions(
        async () => {
            for (const { branchActivity, probInput, sopBuilder } of branchEditors) {
                const body = {
                    probability: parseFloat(probInput.value) || 0,
                    conditions: sopBuilder.getValue(),
                };
                const resp = await fetch(
                    `/api/${document.body.dataset.configName}/xor-split/${encodeURIComponent(placeId)}/${encodeURIComponent(branchActivity)}`,
                    {
                        method: "PATCH",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body),
                    }
                );
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.error || resp.statusText);
                }
                currentData.xor_splits[placeId].branches[branchActivity].probability = body.probability;
                currentData.xor_splits[placeId].branches[branchActivity].conditions = body.conditions;
            }
            if (typeof setDomainStale === "function") setDomainStale(true);
            showXorSplitPanel(placeId);
        },
        () => showXorSplitPanel(placeId)
    ));
}


function _makeEditSection(title, icon) {
    const section = document.createElement("div");
    section.className = "panel-edit-section";
    const label = document.createElement("div");
    label.className = "panel-edit-label";
    if (icon) label.innerHTML = `<i class="bi ${icon}"></i> `;
    label.appendChild(document.createTextNode(title));
    section.appendChild(label);
    return section;
}

function _buildEditActions(onSave, onCancel) {
    const actions = document.createElement("div");
    actions.className = "panel-form-actions";

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "btn-cancel-edit";
    cancelBtn.textContent = "Cancel";
    cancelBtn.addEventListener("click", onCancel);

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "btn-save-edit";
    saveBtn.innerHTML = '<i class="bi bi-check-lg"></i> Save';
    saveBtn.addEventListener("click", async () => {
        saveBtn.disabled = true;
        saveBtn.innerHTML = '<span class="spinner"></span> Saving...';
        try {
            await onSave();
        } catch (err) {
            alert("Save failed: " + err.message);
            saveBtn.disabled = false;
            saveBtn.innerHTML = '<i class="bi bi-check-lg"></i> Save';
        }
    });

    actions.appendChild(cancelBtn);
    actions.appendChild(saveBtn);
    return actions;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function createSection(title, isMeta) {
    const section = document.createElement("div");
    section.className = "panel-section";

    const titleEl = document.createElement("div");
    titleEl.className = `section-title${isMeta ? " meta" : ""}`;

    const icon = SECTION_ICONS[title];
    if (icon) titleEl.innerHTML = `<i class="bi ${icon}"></i> ${title}`;
    else titleEl.textContent = title;

    section.appendChild(titleEl);
    return section;
}

function conditionPill(cond) {
    const span = document.createElement("span");
    const isNeq = cond.predicate === "<>";
    span.className = `condition-pill ${isNeq ? "condition-pill-neq" : "condition-pill-eq"}`;
    span.textContent = `${cond.attribute} ${cond.predicate} ${cond.value}`;
    return span;
}

function orSeparator() {
    const sep = document.createElement("div");
    sep.className = "or-separator";
    sep.textContent = "OR";
    return sep;
}

function cascadeBadge(level) {
    return `<span class="cascade-badge cascade-${level}">${["", "DT", "stat.", "none"][level] || level}</span>`;
}

function emptyNote(text) {
    const p = document.createElement("p");
    p.className = "empty-note";
    p.textContent = text;
    return p;
}

/**
 * Render a SOP (Sum-of-Products) condition block: groups of AND pills
 * separated by OR dividers. Appends directly to the given container.
 *
 * @param {HTMLElement} container - Parent element to append condition groups into
 * @param {Array<Array<Object>>} conditionsSOP - Outer array = OR, inner array = AND
 */
function renderConditionGroups(container, conditionsSOP) {
    for (let i = 0; i < conditionsSOP.length; i++) {
        if (i > 0) container.appendChild(orSeparator());
        const group = document.createElement("div");
        group.className = "condition-group";
        for (const cond of conditionsSOP[i]) {
            group.appendChild(conditionPill(cond));
        }
        container.appendChild(group);
    }
}
