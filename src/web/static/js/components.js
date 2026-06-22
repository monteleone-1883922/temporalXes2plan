/**
 * Shared form components used by monitor.js and panels.js.
 *
 * createSopBuilder  — OR-of-ANDs condition editor (with optional predicate select)
 * createAssignmentList — flat attribute=value list (no predicate, no OR clauses)
 */

// ---------------------------------------------------------------------------
// SOP builder
// ---------------------------------------------------------------------------

/**
 * @param {Object} catalog       - {attr: {type, possible_values}}
 * @param {Array}  initialSop    - [[{attribute, predicate, value}], ...]
 * @param {Object} opts          - {withPredicate: bool (default true)}
 * @returns {{el, getValue, addClause}}
 */
function createSopBuilder(catalog, initialSop = [], opts = {}) {
    const withPredicate = opts.withPredicate !== false;
    // startEmpty: true → no initial clause (use when the field is legitimately empty)
    // startEmpty: false (default) → one empty clause pre-created (use for goal/required fields)
    const startEmpty = opts.startEmpty === true;

    const el = document.createElement("div");
    el.className = "sop-builder";

    const clauseContainer = document.createElement("div");
    clauseContainer.className = "sop-clauses";
    el.appendChild(clauseContainer);

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn-add-clause";
    addBtn.innerHTML = '<i class="bi bi-plus-circle"></i> Add clause';
    addBtn.addEventListener("click", () =>
        _sopAddClause(clauseContainer, catalog, [], withPredicate)
    );
    el.appendChild(addBtn);

    if (initialSop && initialSop.length > 0) {
        for (const andClause of initialSop) {
            _sopAddClause(clauseContainer, catalog, andClause, withPredicate);
        }
    } else if (!startEmpty) {
        _sopAddClause(clauseContainer, catalog, [], withPredicate);
    }

    function getValue() {
        const sop = [];
        clauseContainer.querySelectorAll(":scope > .sop-clause").forEach(clause => {
            const conditions = [];
            clause.querySelectorAll(".sop-cond-row").forEach(row => {
                const attr = row.querySelector(".sop-attr-sel")?.value;
                const predicate = withPredicate
                    ? (row.querySelector(".sop-op-sel")?.value || "=")
                    : "=";
                const value = row.querySelector(".sop-val-sel")?.value;
                if (attr && value !== undefined) conditions.push({ attribute: attr, predicate, value });
            });
            if (conditions.length > 0) sop.push(conditions);
        });
        return sop;
    }

    function addClause() {
        _sopAddClause(clauseContainer, catalog, [], withPredicate);
    }

    return { el, getValue, addClause };
}

function _sopAddClause(clauseContainer, catalog, initialConds, withPredicate) {
    const clause = document.createElement("div");
    clause.className = "sop-clause";

    const header = document.createElement("div");
    header.className = "sop-clause-header";

    const orLabel = document.createElement("span");
    orLabel.className = "sop-or-label";
    orLabel.textContent = clauseContainer.querySelectorAll(".sop-clause").length > 0 ? "OR" : "";
    header.appendChild(orLabel);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-remove-clause";
    removeBtn.innerHTML = '<i class="bi bi-trash3"></i>';
    removeBtn.addEventListener("click", () => {
        clause.remove();
        _sopRefreshOrLabels(clauseContainer);
    });
    header.appendChild(removeBtn);
    clause.appendChild(header);

    const condList = document.createElement("div");
    condList.className = "sop-conditions";
    clause.appendChild(condList);

    const footRow = document.createElement("div");
    footRow.className = "sop-clause-foot";
    const addCondBtn = document.createElement("button");
    addCondBtn.type = "button";
    addCondBtn.className = "btn-add-cond";
    addCondBtn.innerHTML = '<i class="bi bi-plus"></i> Add condition';
    addCondBtn.addEventListener("click", () =>
        _sopAddCondRow(condList, catalog, null, withPredicate)
    );
    footRow.appendChild(addCondBtn);
    clause.appendChild(footRow);

    clauseContainer.appendChild(clause);
    _sopRefreshOrLabels(clauseContainer);

    if (initialConds && initialConds.length > 0) {
        for (const cond of initialConds) {
            _sopAddCondRow(condList, catalog, cond, withPredicate);
        }
    } else {
        _sopAddCondRow(condList, catalog, null, withPredicate);
    }
}

function _sopRefreshOrLabels(clauseContainer) {
    clauseContainer.querySelectorAll(":scope > .sop-clause").forEach((c, i) => {
        const label = c.querySelector(".sop-or-label");
        if (label) label.textContent = i === 0 ? "" : "OR";
    });
}

function _sopAddCondRow(condList, catalog, initialCond, withPredicate) {
    const attrs = Object.keys(catalog).sort();
    if (attrs.length === 0) return;

    const row = document.createElement("div");
    row.className = "sop-cond-row";

    const attrSel = document.createElement("select");
    attrSel.className = "sop-attr-sel";
    for (const attr of attrs) {
        const opt = document.createElement("option");
        opt.value = attr;
        opt.textContent = attr;
        attrSel.appendChild(opt);
    }
    if (initialCond?.attribute) attrSel.value = initialCond.attribute;
    row.appendChild(attrSel);

    if (withPredicate) {
        const opSel = document.createElement("select");
        opSel.className = "sop-op-sel";
        opSel.innerHTML = `<option value="=">=</option><option value="<>">&ne;</option>`;
        if (initialCond?.predicate) opSel.value = initialCond.predicate;
        row.appendChild(opSel);
    }

    const valueWrap = document.createElement("div");
    valueWrap.className = "sop-value-wrap";

    function rebuildValue() {
        const info = catalog[attrSel.value] || {};
        valueWrap.innerHTML = "";
        const sel = document.createElement("select");
        sel.className = "sop-val-sel";
        if (info.type === "boolean") {
            sel.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
        } else {
            for (const v of (info.possible_values || [])) {
                const opt = document.createElement("option");
                opt.value = v;
                opt.textContent = v;
                sel.appendChild(opt);
            }
        }
        if (initialCond?.value !== undefined) sel.value = String(initialCond.value);
        valueWrap.appendChild(sel);
    }
    attrSel.addEventListener("change", rebuildValue);
    rebuildValue();

    row.appendChild(valueWrap);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-remove-cond";
    removeBtn.innerHTML = '<i class="bi bi-x"></i>';
    removeBtn.addEventListener("click", () => row.remove());
    row.appendChild(removeBtn);

    condList.appendChild(row);
}

// ---------------------------------------------------------------------------
// Assignment list (flat attribute=value, no OR clauses, no predicate)
// ---------------------------------------------------------------------------

/**
 * @param {Object} catalog              - {attr: {type, possible_values}}
 * @param {Array}  initialAssignments   - [{attribute, value}]
 * @returns {{el, getValue}}
 */
function createAssignmentList(catalog, initialAssignments = []) {
    const el = document.createElement("div");
    el.className = "assignment-list";

    const rowList = document.createElement("div");
    rowList.className = "assign-rows";
    el.appendChild(rowList);

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn-add-clause";
    addBtn.innerHTML = '<i class="bi bi-plus-circle"></i> Add condition';
    addBtn.addEventListener("click", () => _assignAddRow(rowList, catalog, null));
    el.appendChild(addBtn);

    for (const assignment of (initialAssignments || [])) {
        _assignAddRow(rowList, catalog, assignment);
    }

    function getValue() {
        const assignments = [];
        rowList.querySelectorAll(".sop-cond-row").forEach(row => {
            const attr = row.querySelector(".sop-attr-sel")?.value;
            const value = row.querySelector(".sop-val-sel")?.value;
            if (attr && value !== undefined) assignments.push({ attribute: attr, value });
        });
        return assignments;
    }

    return { el, getValue };
}

function _assignAddRow(rowList, catalog, initialAssignment) {
    const attrs = Object.keys(catalog).sort();
    if (attrs.length === 0) return;

    const row = document.createElement("div");
    row.className = "sop-cond-row";

    const attrSel = document.createElement("select");
    attrSel.className = "sop-attr-sel";
    for (const attr of attrs) {
        const opt = document.createElement("option");
        opt.value = attr;
        opt.textContent = attr;
        attrSel.appendChild(opt);
    }
    if (initialAssignment?.attribute) attrSel.value = initialAssignment.attribute;

    const valueWrap = document.createElement("div");
    valueWrap.className = "sop-value-wrap";

    function rebuildValue() {
        const info = catalog[attrSel.value] || {};
        valueWrap.innerHTML = "";
        const sel = document.createElement("select");
        sel.className = "sop-val-sel";
        if (info.type === "boolean") {
            sel.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
        } else {
            for (const v of (info.possible_values || [])) {
                const opt = document.createElement("option");
                opt.value = v;
                opt.textContent = v;
                sel.appendChild(opt);
            }
        }
        if (initialAssignment?.value !== undefined) sel.value = String(initialAssignment.value);
        valueWrap.appendChild(sel);
    }
    attrSel.addEventListener("change", rebuildValue);
    rebuildValue();

    row.appendChild(attrSel);
    row.appendChild(valueWrap);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-remove-cond";
    removeBtn.innerHTML = '<i class="bi bi-x"></i>';
    removeBtn.addEventListener("click", () => row.remove());
    row.appendChild(removeBtn);

    rowList.appendChild(row);
}

// ---------------------------------------------------------------------------
// Effect group editor — OR of groups, each with assignments + guard + probability
// ---------------------------------------------------------------------------

/**
 * Editor for effect_groups: [{assignments:[{attribute,value}], guard:[...], probability:float}]
 *
 * Each group card exposes:
 *   - a probability input
 *   - a guard SOP builder (reuses createSopBuilder)
 *   - an assignment list where attr is chosen from the full catalog (minus
 *     attrs already used by other rows in the same group)
 *
 * @param {Object} catalog        - {attr: {type, possible_values}}
 * @param {Array}  initialGroups  - [{assignments, guard, probability}, ...]
 * @param {Object} effectsLookup  - {attr: {value: {...}}} (unused in editor, kept for API compat)
 * @returns {{el, getValue}}
 */
function createEffectGroupEditor(catalog, initialGroups, effectsLookup) {
    const el = document.createElement("div");
    el.className = "sop-builder";

    const clauseContainer = document.createElement("div");
    clauseContainer.className = "sop-clauses";
    el.appendChild(clauseContainer);

    const addGroupBtn = document.createElement("button");
    addGroupBtn.type = "button";
    addGroupBtn.className = "btn-add-clause";
    addGroupBtn.innerHTML = '<i class="bi bi-plus-circle"></i> Add group';
    addGroupBtn.addEventListener("click", () => _egAddGroup(clauseContainer, catalog, {}));
    el.appendChild(addGroupBtn);

    for (const group of (initialGroups || [])) {
        _egAddGroup(clauseContainer, catalog, group);
    }

    function getValue() {
        const groups = [];
        clauseContainer.querySelectorAll(":scope > .sop-clause").forEach(clause => {
            const assignments = [];
            clause.querySelector(".eg-assign-rows")
                  ?.querySelectorAll(":scope > .eg-assign-row")
                  .forEach(row => {
                      const attr = row.querySelector(".sop-attr-sel")?.value;
                      const value = row.querySelector(".sop-val-sel")?.value;
                      if (attr && value !== undefined) assignments.push({ attribute: attr, value });
                  });
            if (assignments.length === 0) return;
            groups.push({
                assignments,
                guard: clause._guardBuilder ? clause._guardBuilder.getValue() : [],
                probability: parseFloat(clause._probInput?.value) || 0,
            });
        });
        return groups;
    }

    return { el, getValue };
}

function _egAddGroup(clauseContainer, catalog, initialGroup) {
    const clause = document.createElement("div");
    clause.className = "sop-clause";

    // ── OR label + remove ────────────────────────────────────────────────────
    const header = document.createElement("div");
    header.className = "sop-clause-header";

    const orLabel = document.createElement("span");
    orLabel.className = "sop-or-label";
    orLabel.textContent = clauseContainer.querySelectorAll(".sop-clause").length > 0 ? "OR" : "";
    header.appendChild(orLabel);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-remove-clause";
    removeBtn.innerHTML = '<i class="bi bi-trash3"></i>';
    removeBtn.addEventListener("click", () => {
        clause.remove();
        _sopRefreshOrLabels(clauseContainer);
    });
    header.appendChild(removeBtn);
    clause.appendChild(header);

    // ── Probability ──────────────────────────────────────────────────────────
    const probRow = document.createElement("div");
    probRow.style.cssText = "display:flex;align-items:center;gap:8px;padding:4px 0 6px 0;";
    const probLabel = document.createElement("span");
    probLabel.className = "meta-key";
    probLabel.textContent = "Probability";
    const probInput = document.createElement("input");
    probInput.type = "number";
    probInput.className = "prob-input";
    probInput.min = "0";
    probInput.max = "1";
    probInput.step = "0.0001";
    probInput.value = initialGroup.probability ?? 1;
    probRow.appendChild(probLabel);
    probRow.appendChild(probInput);
    clause.appendChild(probRow);
    clause._probInput = probInput;

    // ── Guard SOP builder ────────────────────────────────────────────────────
    const guardLabel = document.createElement("div");
    guardLabel.className = "cond-label";
    guardLabel.textContent = "Guard";
    clause.appendChild(guardLabel);
    const guardBuilder = createSopBuilder(catalog, initialGroup.guard || [], {
        withPredicate: true,
        startEmpty: !(initialGroup.guard?.length),
    });
    clause.appendChild(guardBuilder.el);
    clause._guardBuilder = guardBuilder;

    // ── Assignments ──────────────────────────────────────────────────────────
    const assignLabel = document.createElement("div");
    assignLabel.className = "cond-label";
    assignLabel.style.marginTop = "6px";
    assignLabel.textContent = "Assignments";
    clause.appendChild(assignLabel);

    const rowList = document.createElement("div");
    rowList.className = "sop-conditions eg-assign-rows";
    clause.appendChild(rowList);

    const foot = document.createElement("div");
    foot.className = "sop-clause-foot";
    const addRowBtn = document.createElement("button");
    addRowBtn.type = "button";
    addRowBtn.className = "btn-add-cond";
    addRowBtn.innerHTML = '<i class="bi bi-plus"></i> Add assignment';
    addRowBtn.addEventListener("click", () => _egAddRow(rowList, catalog, null));
    foot.appendChild(addRowBtn);
    clause.appendChild(foot);

    clauseContainer.appendChild(clause);
    _sopRefreshOrLabels(clauseContainer);

    const assignments = initialGroup.assignments || [];
    if (assignments.length > 0) {
        for (const a of assignments) _egAddRow(rowList, catalog, a);
    } else {
        _egAddRow(rowList, catalog, null);
    }
}

/**
 * Rebuild each row's attribute select so that attrs used by sibling rows are
 * excluded, preserving the current selection when it remains valid.
 */
function _egRefreshGroupAttrs(rowList, catalog) {
    const allAttrs = Object.keys(catalog).sort();
    const rows = [...rowList.querySelectorAll(":scope > .eg-assign-row")];
    for (const row of rows) {
        const sel = row.querySelector(".sop-attr-sel");
        if (!sel) continue;
        const currentVal = sel.value;
        const otherUsed = new Set(
            rows.filter(r => r !== row)
                .map(r => r.querySelector(".sop-attr-sel")?.value)
                .filter(Boolean)
        );
        sel.innerHTML = "";
        for (const attr of allAttrs) {
            if (!otherUsed.has(attr)) sel.appendChild(new Option(attr, attr));
        }
        if ([...sel.options].some(o => o.value === currentVal)) sel.value = currentVal;
    }
}

function _egAddRow(rowList, catalog, initialAssignment) {
    const allAttrs = Object.keys(catalog).sort();
    if (allAttrs.length === 0) return;

    const row = document.createElement("div");
    row.className = "sop-cond-row eg-assign-row";

    const attrSel = document.createElement("select");
    attrSel.className = "sop-attr-sel";
    for (const attr of allAttrs) {
        attrSel.appendChild(new Option(attr, attr));
    }
    if (initialAssignment?.attribute) attrSel.value = initialAssignment.attribute;

    const arrowSpan = document.createElement("span");
    arrowSpan.className = "effect-eq-label";
    arrowSpan.style.cssText = "padding:0 4px;color:var(--slate-400);";
    arrowSpan.textContent = "→";

    const valueWrap = document.createElement("div");
    valueWrap.className = "sop-value-wrap";

    function rebuildValue() {
        const info = catalog[attrSel.value] || {};
        valueWrap.innerHTML = "";
        const sel = document.createElement("select");
        sel.className = "sop-val-sel";
        if (info.type === "boolean") {
            sel.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
        } else {
            for (const v of (info.possible_values || [])) {
                sel.appendChild(new Option(v, v));
            }
        }
        if (initialAssignment?.value !== undefined) sel.value = String(initialAssignment.value);
        valueWrap.appendChild(sel);
    }

    attrSel.addEventListener("change", () => {
        rebuildValue();
        _egRefreshGroupAttrs(rowList, catalog);
    });
    rebuildValue();

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-remove-cond";
    removeBtn.innerHTML = '<i class="bi bi-x"></i>';
    removeBtn.addEventListener("click", () => {
        row.remove();
        _egRefreshGroupAttrs(rowList, catalog);
    });

    row.appendChild(attrSel);
    row.appendChild(arrowSpan);
    row.appendChild(valueWrap);
    row.appendChild(removeBtn);

    rowList.appendChild(row);
    _egRefreshGroupAttrs(rowList, catalog);
}
