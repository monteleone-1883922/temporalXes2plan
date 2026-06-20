/**
 * Detail panels for transitions and XOR splits — styled version.
 */

let currentData = null;

function setCurrentData(data) { currentData = data; }

function closePanel() {
    document.getElementById("detail-panel").classList.add("d-none");
    document.getElementById("panel-content").innerHTML = "";
    if (cy) cy.nodes(":selected").unselect();
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
    if (t.duration) content.appendChild(buildDurationSection(t.duration));
    content.appendChild(buildPreconditionsSection(t.preconditions));
    content.appendChild(buildEffectsSection(t.effects));
    content.appendChild(buildCostSection(t.cost));

    document.getElementById("detail-panel").classList.remove("d-none");
}

function buildMetaSection(t) {
    const section = createSection("Analysis Info", true);
    const grid = document.createElement("div");
    grid.className = "meta-block";

    const rows = [
        ["Firings", t._meta.total_firings],
    ];

    if (t._meta.xor_branch) {
        const xb = t._meta.xor_branch;
        rows.push(
            ["XOR prob.", `${(xb.probability * 100).toFixed(1)}%`],
            ["Cascade", cascadeBadge(xb.cascade_level)],
            ["Samples", xb.total_samples],
        );
    }
    if (t._meta.related_effects.length > 0) {
        rows.push(["Related", t._meta.related_effects.map(p => p.join(" & ")).join(", ")]);
    }
    if (t._meta.incompatible_effects.length > 0) {
        rows.push(["Incompatible", t._meta.incompatible_effects.map(p => p.join(" & ")).join(", ")]);
    }

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

    for (let i = 0; i < preconditions.length; i++) {
        if (i > 0) section.appendChild(orSeparator());
        const group = document.createElement("div");
        group.className = "condition-group";
        for (const cond of preconditions[i]) group.appendChild(conditionPill(cond));
        section.appendChild(group);
    }

    return section;
}

function buildEffectsSection(effects) {
    const section = createSection("Effects");

    if (!effects || effects.length === 0) {
        section.appendChild(emptyNote("No effects"));
        return section;
    }

    for (const eff of effects) {
        const card = document.createElement("div");
        card.className = "effect-card";

        // Header row: attribute=value + probability badge
        const headerRow = document.createElement("div");
        headerRow.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;";
        const hdr = document.createElement("span");
        hdr.className = "effect-header";
        hdr.textContent = `${eff.attribute} = ${eff.value}`;
        const badge = document.createElement("span");
        badge.className = "prob-badge";
        badge.textContent = `${(eff.probability * 100).toFixed(1)}%`;
        headerRow.appendChild(hdr);
        headerRow.appendChild(badge);
        card.appendChild(headerRow);

        // Effect preconditions
        if (eff.preconditions && eff.preconditions.length > 0) {
            const label = document.createElement("div");
            label.className = "cond-label";
            label.textContent = "Conditions";
            card.appendChild(label);
            for (let i = 0; i < eff.preconditions.length; i++) {
                if (i > 0) card.appendChild(orSeparator());
                const group = document.createElement("div");
                group.className = "condition-group";
                for (const cond of eff.preconditions[i]) group.appendChild(conditionPill(cond));
                card.appendChild(group);
            }
        }

        // Meta row
        const meta = document.createElement("div");
        meta.className = "effect-meta";
        meta.innerHTML = `
            <span class="meta-key">Presence</span>
            <span>${(eff._meta.presence_probability * 100).toFixed(1)}%</span>
            <span style="color:#cbd5e1">·</span>
            <span class="meta-key">App.</span> ${cascadeBadge(eff._meta.appearance_level)}
            <span style="color:#cbd5e1">·</span>
            <span class="meta-key">Val.</span> ${cascadeBadge(eff._meta.value_level)}
        `;
        card.appendChild(meta);

        section.appendChild(card);
    }

    return section;
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

    for (const branch of xor.branches) {
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
            for (let i = 0; i < branch.conditions.length; i++) {
                if (i > 0) card.appendChild(orSeparator());
                const group = document.createElement("div");
                group.className = "condition-group";
                for (const cond of branch.conditions[i]) group.appendChild(conditionPill(cond));
                card.appendChild(group);
            }
        } else {
            card.appendChild(emptyNote("No guards — statistical fallback"));
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

    document.getElementById("detail-panel").classList.remove("d-none");
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
