/**
 * Cytoscape.js graph rendering for Petri nets.
 */

let cy = null;

function initGraph() {
    cy = cytoscape({
        container: document.getElementById("cy"),
        style: [
            // ── Base node ──────────────────────────────────────────────────
            {
                selector: "node",
                style: {
                    label: "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "11px",
                    "font-family": "Inter, system-ui, sans-serif",
                    "font-weight": "500",
                    "text-wrap": "ellipsis",
                    "text-max-width": "110px",
                    "color": "#1e293b",
                },
            },

            // ── Place ──────────────────────────────────────────────────────
            {
                selector: 'node[type="place"]',
                style: {
                    shape: "ellipse",
                    "background-color": "#ffffff",
                    "border-color": "#475569",
                    "border-width": 2,
                    width: 42,
                    height: 42,
                    "font-size": "10px",
                    color: "#475569",
                    "text-valign": "bottom",
                    "text-margin-y": 4,
                },
            },

            // ── XOR split ──────────────────────────────────────────────────
            {
                selector: 'node[type="xor_split"]',
                style: {
                    shape: "ellipse",
                    "background-color": "#fef3c7",
                    "border-color": "#f59e0b",
                    "border-width": 2.5,
                    "border-style": "dashed",
                    width: 50,
                    height: 50,
                    "font-size": "10px",
                    color: "#92400e",
                    "font-weight": "700",
                    "text-valign": "bottom",
                    "text-margin-y": 4,
                },
            },

            // ── Labeled transition ─────────────────────────────────────────
            {
                selector: 'node[type="transition"]',
                style: {
                    shape: "round-rectangle",
                    "background-color": "#eef2ff",
                    "border-color": "#6366f1",
                    "border-width": 1.5,
                    "corner-radius": 8,
                    width: "label",
                    height: 34,
                    "padding-left": "14px",
                    "padding-right": "14px",
                    "font-size": "11px",
                    "font-weight": "600",
                    color: "#3730a3",
                },
            },

            // ── AND split/join ─────────────────────────────────────────────
            {
                selector: 'node[type="and_split"]',
                style: {
                    shape: "round-rectangle",
                    "background-color": "#dcfce7",
                    "border-color": "#16a34a",
                    "border-width": 2,
                    "corner-radius": 8,
                    width: "label",
                    height: 34,
                    "padding-left": "14px",
                    "padding-right": "14px",
                    "font-size": "11px",
                    "font-weight": "700",
                    color: "#14532d",
                },
            },

            // ── Silent (tau) ───────────────────────────────────────────────
            {
                selector: 'node[type="silent"]',
                style: {
                    shape: "rectangle",
                    "background-color": "#64748b",
                    "border-color": "#475569",
                    "border-width": 1,
                    width: 10,
                    height: 32,
                    label: "",
                },
            },

            // ── Edges ──────────────────────────────────────────────────────
            {
                selector: "edge",
                style: {
                    width: 1.5,
                    "line-color": "#94a3b8",
                    "target-arrow-color": "#94a3b8",
                    "target-arrow-shape": "triangle",
                    "arrow-scale": 1.0,
                    "curve-style": "bezier",
                },
            },

            // ── Hover ──────────────────────────────────────────────────────
            {
                selector: "node:active",
                style: {
                    "overlay-color": "#6366f1",
                    "overlay-opacity": 0.12,
                    "overlay-padding": 4,
                },
            },

            // ── Selected ───────────────────────────────────────────────────
            {
                selector: "node:selected",
                style: {
                    "border-color": "#6366f1",
                    "border-width": 3,
                    "overlay-color": "#6366f1",
                    "overlay-opacity": 0.1,
                    "overlay-padding": 4,
                },
            },
        ],
        layout: { name: "preset" },
        wheelSensitivity: 0.3,
        minZoom: 0.2,
        maxZoom: 3,
    });

    cy.on("tap", "node", function (evt) {
        const node  = evt.target;
        const type  = node.data("type");
        const id    = node.data("id");
        const label = node.data("label");

        if (type === "place") {
            onPlaceSelected(id);
            closePanel();
        } else if (type === "transition" || type === "and_split") {
            onPlaceClear();
            showTransitionPanel(label);
        } else if (type === "xor_split") {
            onPlaceSelected(id);
            showXorSplitPanel(id);
        } else {
            onPlaceClear();
            closePanel();
        }
    });

    cy.on("tap", function (evt) {
        if (evt.target === cy) {
            closePanel();
            onPlaceClear();
        }
    });
}

function renderGraph(data) {
    if (!cy) initGraph();
    cy.elements().remove();

    const elements = [];

    for (const node of data.graph.nodes) {
        elements.push({
            group: "nodes",
            data: {
                id:           node.id,
                label:        node.label,
                type:         node.type,
                is_silent:    node.is_silent || false,
            },
        });
    }

    for (const edge of data.graph.edges) {
        elements.push({
            group: "edges",
            data: {
                id:     edge.source + "->" + edge.target,
                source: edge.source,
                target: edge.target,
            },
        });
    }

    cy.add(elements);

    cy.layout({
        name: "dagre",
        rankDir: "LR",
        nodeSep: 44,
        rankSep: 90,
        edgeSep: 20,
        animate: false,
    }).run();

    cy.fit(undefined, 40);
}
