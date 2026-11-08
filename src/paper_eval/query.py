"""Query every result of the paper evaluation with SQL (DuckDB, in memory, no server).

Nothing is converted here. Each phase writes its output in a shape DuckDB reads
as it is (one row per entity, see paper_eval.model_tables), and this module only
declares views over those files. The files stay the only source of truth and
nothing is written. A model run is (log, split_hash) = the folder
results/<log>/<split_hash>/; every table carries those two columns, so several
logs can live in the same database (`connect_all`).

Tables (the file each one reads, inside the run folder unless noted):

  runs                  one row per run                  split.json, prefixes_meta.json, model_meta.json
  logs                  one row per log                  <data_dir>/<log>.stats.json
  case_split            one row per case; part =         split.json (the id lists)
                        train_fit / val / test, or
                        removed_spanning (taken out of
                        TRAIN: still running at the split)
  models                one row per run: net, attributes,   model_stats.json
                        effect groups, PDDL, durations
  search_trials         one row per TPE trial            search_trials.json
  attributes            one row per attribute of the log    attributes.jsonl
                        and its fate: status = kept /
                        removed_unused / removed_noise /
                        removed_cascade / ignored_by_user
  transition_durations  one row per transition           transition_durations.jsonl
  prefixes              one row per (test case, k, goal) prefixes.jsonl
  instances             one row per deduplicated instance   instances.jsonl
  plan_meta             one row per run: OPTIC settings     plan_meta.json
                        and environment
  plan_results          one row per instance: OPTIC         plan_results.jsonl
                        outcome and times
  rq2_meta              one row per run: RQ2 settings       rq2_meta.json
  rq2_results           one row per completion instance:    rq2_results.jsonl
                        RQ2 outcome, search_completed
                        (OPTIC ended by itself), best makespan
  rq2_plans             one row per plan OPTIC printed      rq2_plans.jsonl
  rq2_plan_steps        one row per step of every plan      rq2_plan_steps.jsonl

`logs`, `models` and `plan_meta` keep the nesting of their files as STRUCT
columns: `SELECT petri_net.places, pddl.actions FROM models`,
`SELECT trace_length.max FROM logs`.

Views derived from them:

  cases           prefixes LEFT JOIN plan_results on (log, split_hash, group_id):
                  one row per (case, k, goal) with the outcome of its instance
                  (status NULL if not run), plus fits_deadline_best_case and
                  fits_deadline_worst_case (deadline goal only, see below).
  instances_view  one row per instance: n_cases, n_cases_replayable,
                  replayability ('all' / 'mixed' / 'none') plus the plan columns.

  rq2_reference   one row per instance with a best plan: its activities in the order of
                  their start time (silent steps left out, same-start steps in print order),
                  the test cases that start with the same prefix, how many continue with
                  exactly that sequence (n_matching) and the one with the shortest remaining
                  time (reference_case_id, reference_rt_s, NULL when none matches). Times in
                  seconds; differences and ratios are left to the query:
                  `SELECT best_makespan_s - reference_rt_s FROM rq2_reference`.

Keys: runs(log, split_hash) is the hub. prefixes, instances and plan_results
join on group_id inside a run; prefixes.case_id points to case_split.

A file that does not exist (a phase not run yet) gives an empty table with the
right columns, except prefixes.jsonl, without which there is nothing to query.

Attributes. `n_values` is the number of distinct labels in `possible_values`,
the values that appear in the effects and guards of the model (not every value
of the log); numerical attributes have the interval labels of the discretization
plus "none". `bin_boundaries` are the cut points of the Discretizer (KDE and
Ckmeans clustering); a numerical attribute with n cut points has n + 1 intervals.

Weighting. Several cases share one instance and so one OPTIC run. Averaging a
time over `cases` counts that run once per case; averaging over `instances_view`
counts it once. `rq1_times` lets you pick (weight="instance" or "case").

fits_deadline_*: for a deadline record, whether the model's replay can reach
the end within the record's deadline: reached_end AND min (best case) or max
(worst case) modeled duration of the tail <= deadline_s. OPTIC chooses each
duration inside [min, max], so the best case is the one that decides feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb

PLAN_COLUMNS = (
    "status", "grounding_s", "planning_s", "first_plan_s", "wall_s", "cpu_s",
    "states_evaluated", "makespan_s", "plan_length", "timeout_s", "memory_mb", "message",
)
TIME_FIELDS = ("grounding_s", "planning_s", "wall_s")
_PLAN_TEXT_COLUMNS = {"status", "message"}

RUN_KEYS = ("log", "split_hash")
LOG_KEYS = ("log",)

# (column, SQL type) of the files a run may not have yet: the empty table that
# stands in for a missing file. Files not listed (models, plan_meta, logs) are
# empty with only their key columns.
_Columns = Tuple[Tuple[str, str], ...]
_EMPTY: Dict[str, _Columns] = {
    "split.json": (
        ("log_sha256", "VARCHAR"),
        ("params", "STRUCT(test_pct DOUBLE, val_pct DOUBLE)"),
        ("train_fit_ids", "VARCHAR[]"), ("val_ids", "VARCHAR[]"), ("test_ids", "VARCHAR[]"),
        ("removed_spanning_ids", "VARCHAR[]"),
    ),
    "prefixes_meta.json": (("deadline_variant_only", "BOOLEAN"),),
    "model_meta.json": (("discovery_algorithm", "VARCHAR"), ("seed", "BIGINT"), ("unit_costs", "BOOLEAN")),
    "search_trials.json": (
        ("trial_number", "BIGINT"), ("score", "DOUBLE"),
        ("rejected_reason", "VARCHAR"), ("max_effect_groups", "BIGINT"),
    ),
    "attributes.jsonl": (
        ("attribute", "VARCHAR"), ("type", "VARCHAR"), ("status", "VARCHAR"), ("reason", "VARCHAR"),
        ("kept_by_user", "BOOLEAN"), ("n_values", "BIGINT"), ("possible_values", "VARCHAR[]"),
        ("n_bin_boundaries", "BIGINT"), ("bin_boundaries", "DOUBLE[]"), ("n_distinct_observed", "BIGINT"),
    ),
    "transition_durations.jsonl": (
        ("transition", "VARCHAR"), ("activity_name", "VARCHAR"), ("n_effect_groups", "BIGINT"),
        ("cost", "DOUBLE"), ("has_duration", "BOOLEAN"), ("effective_min", "DOUBLE"),
        ("effective_max", "DOUBLE"), ("mean", "DOUBLE"), ("std_dev", "DOUBLE"),
        ("observed_min", "DOUBLE"), ("observed_max", "DOUBLE"), ("count", "BIGINT"), ("source", "VARCHAR"),
    ),
    "instances.jsonl": (
        ("group_id", "BIGINT"), ("goal", "VARCHAR"), ("k", "BIGINT"), ("bucket", "VARCHAR"),
    ),
    "plan_results.jsonl": (("group_id", "BIGINT"),) + tuple(
        (c, "VARCHAR" if c in _PLAN_TEXT_COLUMNS else "DOUBLE") for c in PLAN_COLUMNS
    ),
    "rq2_results.jsonl": (
        ("group_id", "BIGINT"), ("k", "BIGINT"), ("bucket", "VARCHAR"), ("prefix_acts", "VARCHAR[]"),
        ("n_cases", "BIGINT"), ("timeout_s", "BIGINT"), ("memory_mb", "BIGINT"), ("status", "VARCHAR"),
        ("search_completed", "BOOLEAN"), ("n_plans", "BIGINT"), ("plan_length", "BIGINT"),
        ("grounding_s", "DOUBLE"), ("first_plan_s", "DOUBLE"), ("last_plan_s", "DOUBLE"),
        ("wall_s", "DOUBLE"), ("cpu_s", "DOUBLE"), ("best_makespan_s", "DOUBLE"), ("message", "VARCHAR"),
    ),
    "rq2_plans.jsonl": (
        ("group_id", "BIGINT"), ("plan_index", "BIGINT"), ("found_s", "DOUBLE"),
        ("makespan_s", "DOUBLE"), ("is_final", "BOOLEAN"), ("is_best", "BOOLEAN"),
    ),
    "rq2_plan_steps.jsonl": (
        ("group_id", "BIGINT"), ("plan_index", "BIGINT"), ("step_index", "BIGINT"),
        ("action", "VARCHAR"), ("activity", "VARCHAR"), ("is_tau", "BOOLEAN"),
        ("start_s", "DOUBLE"), ("duration_s", "DOUBLE"),
    ),
}

# What rq2_reference needs from the prefixes view (a hand-made prefixes file may lack them).
_RQ2_PREFIX_COLUMNS = {"case_id", "goal", "prefix_acts", "true_suffix_acts", "true_rt_s"}

_RQ2_REFERENCE_COLUMNS = (
    "log VARCHAR, split_hash VARCHAR, group_id BIGINT, k BIGINT, bucket VARCHAR, "
    "prefix_acts VARCHAR[], status VARCHAR, search_completed BOOLEAN, best_makespan_s DOUBLE, "
    "plan_activities VARCHAR[], n_test_cases_same_prefix BIGINT, n_matching BIGINT, "
    "reference_case_id VARCHAR, reference_rt_s DOUBLE"
)

_RQ2_REFERENCE_SQL = """
CREATE VIEW rq2_reference AS
WITH sequences AS (
    SELECT log, split_hash, group_id, plan_index,
           list(activity ORDER BY start_s, step_index) FILTER (WHERE NOT is_tau) AS activities
    FROM rq2_plan_steps GROUP BY log, split_hash, group_id, plan_index
), best AS (
    SELECT r.log, r.split_hash, r.group_id, r.k, r.bucket, r.prefix_acts, r.status,
           r.search_completed, r.best_makespan_s,
           coalesce(s.activities, []::VARCHAR[]) AS plan_activities
    FROM rq2_results r
    JOIN rq2_plans p ON p.log = r.log AND p.split_hash = r.split_hash
                    AND p.group_id = r.group_id AND p.is_best
    LEFT JOIN sequences s ON s.log = p.log AND s.split_hash = p.split_hash
                         AND s.group_id = p.group_id AND s.plan_index = p.plan_index
), same_prefix AS (
    SELECT b.log, b.split_hash, b.group_id, c.case_id, c.true_rt_s,
           (c.true_suffix_acts = b.plan_activities) AS matches
    FROM best b
    JOIN prefixes c ON c.log = b.log AND c.split_hash = b.split_hash
                   AND c.goal = 'completion' AND c.prefix_acts = b.prefix_acts
), summary AS (
    SELECT log, split_hash, group_id,
           count(*) AS n_test_cases_same_prefix,
           count(*) FILTER (WHERE matches) AS n_matching,
           arg_min(case_id, true_rt_s) FILTER (WHERE matches) AS reference_case_id,
           min(true_rt_s) FILTER (WHERE matches) AS reference_rt_s
    FROM same_prefix GROUP BY log, split_hash, group_id
)
SELECT b.log, b.split_hash, b.group_id, b.k, b.bucket, b.prefix_acts, b.status, b.search_completed,
       b.best_makespan_s, b.plan_activities,
       coalesce(s.n_test_cases_same_prefix, 0) AS n_test_cases_same_prefix,
       coalesce(s.n_matching, 0) AS n_matching,
       s.reference_case_id, s.reference_rt_s
FROM best b LEFT JOIN summary s
  ON s.log = b.log AND s.split_hash = b.split_hash AND s.group_id = b.group_id
"""


@dataclass(frozen=True)
class Run:
    """One model run: results/<log>/<split_hash>/."""

    log: str
    split_hash: str
    model_dir: Path


# One file and the values of the key columns it belongs to, e.g.
# ({"log": "sepsis", "split_hash": "87d8..."}, Path(".../search_trials.json")).
Entry = Tuple[Dict[str, str], Path]


# ---------------------------------------------------------------------------
# Reading files with DuckDB
# ---------------------------------------------------------------------------

def _quote(text: Any) -> str:
    """SQL string literal: apostrophes are doubled."""
    return "'" + str(text).replace("'", "''") + "'"


def _usable(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _source(
    con: "duckdb.DuckDBPyConnection", entries: Sequence[Entry], keys: Sequence[str],
    empty: _Columns = (), hide: Iterable[str] = (), tolerate_partial_line: bool = False,
    where: str = "",
) -> str:
    """The text of a SELECT with the rows of all the files in `entries`, each row
    in front of the key columns of the file it came from.

    DuckDB reads the files itself (`read_json_auto`), all in one call:
      union_by_name  files of different runs may differ in columns: merge by name;
      filename       adds each row's source file, to look up its keys;
      sample_size=-1 infer types from every row, not from the first few (a column
                     that starts with NULLs would be typed wrongly).
    The keys come from a VALUES list of (file, key...) joined on that filename.
    When no file exists, a SELECT of the same shape with zero rows (`empty`).

    Args:
        keys: names of the key columns, in the order of each entry's dict.
        hide: columns of the files not to expose.
        tolerate_partial_line: skip lines that are not valid JSON (a line cut
            short by a killed run); DuckDB turns such a line into an all-NULL
            row, which `where` must then drop.
        where: condition added to the SELECT (alias `j` is the file).
    """
    files = [(k, p) for k, p in entries if _usable(p)]
    if not files:
        columns = [f"NULL::VARCHAR AS {k}" for k in keys] + [f"NULL::{t} AS {c}" for c, t in empty]
        return f"SELECT {', '.join(columns)} WHERE false"
    options = "union_by_name=true, filename=true, sample_size=-1"
    if tolerate_partial_line:
        options += ", ignore_errors=true"
    source = f"read_json_auto([{', '.join(_quote(p) for _, p in files)}], {options})"
    # Column list built here, not with SQL lambdas (their syntax changes between versions).
    hidden = {"filename", "file", *keys, *hide}
    names = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
    columns = ", ".join(f'j."{c}"' for c in names if c not in hidden)
    owners = ", ".join(
        "(" + ", ".join(_quote(v) for v in [p, *(k[name] for name in keys)]) + ")" for k, p in files
    )
    select = ", ".join(f"o.{k}" for k in keys) + (f", {columns}" if columns else "")
    return (
        f"SELECT {select} FROM {source} j "
        f"JOIN (VALUES {owners}) AS o(file, {', '.join(keys)}) ON j.filename = o.file"
        + (f" WHERE {where}" if where else "")
    )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def _build(runs: List[Run], data_dir: Optional[Path] = None) -> "duckdb.DuckDBPyConnection":
    if not any(_usable(r.model_dir / "prefixes.jsonl") for r in runs):
        where = runs[0].model_dir if len(runs) == 1 else "any run"
        raise FileNotFoundError(f"{where}/prefixes.jsonl not found or empty: run the 'prefixes' phase first")
    con = duckdb.connect()

    def run_source(filename: str, **options: Any) -> str:
        entries = [({"log": r.log, "split_hash": r.split_hash}, r.model_dir / filename) for r in runs]
        return _source(con, entries, RUN_KEYS, _EMPTY.get(filename, ()), **options)

    def run_view(name: str, filename: str, **options: Any) -> None:
        con.execute(f"CREATE VIEW {name} AS {run_source(filename, **options)}")

    # --- tables that are a file as it is --------------------------------------
    run_view("models", "model_stats.json", hide=("model_dir", "split", "attribute_filter"))
    run_view("search_trials", "search_trials.json")
    run_view("attributes", "attributes.jsonl")
    run_view("transition_durations", "transition_durations.jsonl")
    run_view("prefixes", "prefixes.jsonl")
    run_view("instances", "instances.jsonl")
    run_view("plan_meta", "plan_meta.json")
    # A run killed while writing leaves a half line: skipped, as plan.read_results
    # does. Every real row has a group_id; the all-NULL row DuckDB makes of a
    # broken line does not.
    run_view("plan_results", "plan_results.jsonl", tolerate_partial_line=True,
             where="j.group_id IS NOT NULL")
    # The RQ2 files, written the same way (and read the same way after a killed run).
    run_view("rq2_meta", "rq2_meta.json")
    for name in ("rq2_results", "rq2_plans", "rq2_plan_steps"):
        run_view(name, f"{name}.jsonl", tolerate_partial_line=True, where="j.group_id IS NOT NULL")

    logs = sorted({r.log for r in runs}) if data_dir is not None else []
    log_entries = [({"log": log}, Path(data_dir) / f"{log}.stats.json") for log in logs]
    con.execute(f"CREATE VIEW logs AS {_source(con, log_entries, LOG_KEYS)}")

    # --- runs and case_split: SQL over each run's own split.json ---------------
    con.execute(f"CREATE VIEW _split AS {run_source('split.json')}")
    con.execute(f"CREATE VIEW _prefixes_meta AS {run_source('prefixes_meta.json')}")
    con.execute(f"CREATE VIEW _model_meta AS {run_source('model_meta.json')}")
    # split.json written before the option existed has no removed_spanning_ids: no case removed.
    split_columns = {row[0] for row in con.execute("DESCRIBE _split").fetchall()}
    has_removed = "removed_spanning_ids" in split_columns
    meta_columns = {row[0] for row in con.execute("DESCRIBE _model_meta").fetchall()}
    has_unit_costs = "unit_costs" in meta_columns
    run_rows = ", ".join(f"({_quote(r.log)}, {_quote(r.split_hash)}, {_quote(r.model_dir)})" for r in runs)
    con.execute(
        "CREATE VIEW runs AS SELECT o.log, o.split_hash, o.model_dir, s.log_sha256, "
        "s.params.test_pct AS test_pct, s.params.val_pct AS val_pct, "
        "len(s.train_fit_ids) AS n_train_fit, len(s.val_ids) AS n_val, len(s.test_ids) AS n_test, "
        + ("len(s.removed_spanning_ids)" if has_removed else "NULL::BIGINT") + " AS n_removed_spanning, "
        # A run made before the option existed used the min rule.
        "coalesce(m.deadline_variant_only, false) AS deadline_variant_only, "
        "mm.discovery_algorithm AS discovery_algorithm, mm.seed AS seed, "
        + ("mm.unit_costs" if has_unit_costs else "NULL::BOOLEAN") + " AS unit_costs "
        f"FROM (VALUES {run_rows}) AS o(log, split_hash, model_dir) "
        "LEFT JOIN _split s ON s.log = o.log AND s.split_hash = o.split_hash "
        "LEFT JOIN _prefixes_meta m ON m.log = o.log AND m.split_hash = o.split_hash "
        "LEFT JOIN _model_meta mm ON mm.log = o.log AND mm.split_hash = o.split_hash"
    )
    # One SELECT per part: unnest turns each list of case ids into one row per id.
    case_split = " UNION ALL ".join(
        f"SELECT log, split_hash, CAST(unnest({ids}) AS VARCHAR) AS case_id, '{part}' AS part FROM _split"
        for part, ids in (
            ("train_fit", "train_fit_ids"), ("val", "val_ids"), ("test", "test_ids"),
            *((("removed_spanning", "removed_spanning_ids"),) if has_removed else ()),
        )
    )
    con.execute(f"CREATE VIEW case_split AS {case_split}")

    # --- views derived from prefixes and plan_results ---------------------------
    # A column plan_results.jsonl does not have (an older run, a hand-made file) is NULL.
    in_plan = {row[0] for row in con.execute("DESCRIBE plan_results").fetchall()}
    plan_cols = ", ".join(f"r.{c}" if c in in_plan else f"NULL AS {c}" for c in PLAN_COLUMNS)
    have = {row[0] for row in con.execute("DESCRIBE prefixes").fetchall()}

    def fits(modeled: str) -> str:
        # Prefix files always have these columns; a hand-made file in a test may not.
        if not {"goal", "reached_end", modeled, "deadline_s"} <= have:
            return "NULL::BOOLEAN"
        return f"CASE WHEN p.goal = 'deadline' THEN p.reached_end AND p.{modeled} <= p.deadline_s END"

    con.execute(
        f"CREATE VIEW cases AS SELECT p.*, {plan_cols}, "
        f"{fits('min_modeled_duration_s')} AS fits_deadline_best_case, "
        f"{fits('max_modeled_duration_s')} AS fits_deadline_worst_case "
        "FROM prefixes p LEFT JOIN plan_results r USING (log, split_hash, group_id)"
    )
    # Rows of `cases` collapse to one per instance: the plan columns are the same
    # for every case of an instance, the counts come from the cases.
    replayable = "count(*) FILTER (WHERE coalesce(is_replayable, false))"
    if _RQ2_PREFIX_COLUMNS <= have:
        con.execute(_RQ2_REFERENCE_SQL)
    else:
        columns = ", ".join(
            f"NULL::{definition.split(' ', 1)[1]} AS {definition.split(' ', 1)[0]}"
            for definition in _RQ2_REFERENCE_COLUMNS.split(", ")
        )
        con.execute(f"CREATE VIEW rq2_reference AS SELECT {columns} WHERE false")
    con.execute(
        "CREATE VIEW instances_view AS SELECT "
        "log, split_hash, group_id, any_value(goal) AS goal, any_value(k) AS k, "
        "any_value(bucket) AS bucket, count(*) AS n_cases, "
        f"{replayable} AS n_cases_replayable, "
        f"CASE WHEN {replayable} = count(*) THEN 'all' WHEN {replayable} = 0 THEN 'none' "
        "ELSE 'mixed' END AS replayability, "
        + ", ".join(f"any_value({c}) AS {c}" for c in PLAN_COLUMNS)
        + " FROM cases GROUP BY log, split_hash, group_id"
    )
    return con


def connect(model_dir: Path, data_dir: Optional[Path] = None) -> "duckdb.DuckDBPyConnection":
    """In-memory connection over ONE run (results/<log>/<split_hash>/).
    data_dir (the folder with <log>.stats.json) fills the `logs` table."""
    model_dir = Path(model_dir).resolve()
    return _build([Run(model_dir.parent.name, model_dir.name, model_dir)], data_dir)


def discover_runs(results_dir: Path) -> List[Run]:
    """Every results/<log>/<split_hash>/ that holds a built model."""
    runs: List[Run] = []
    for log_dir in sorted(Path(results_dir).resolve().iterdir()):
        if not log_dir.is_dir():
            continue
        for model_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
            if (model_dir / "current.json").is_file():
                runs.append(Run(log_dir.name, model_dir.name, model_dir))
    return runs


def connect_all(results_dir: Path, data_dir: Optional[Path] = None) -> "duckdb.DuckDBPyConnection":
    """In-memory connection over every run found in results_dir."""
    return _build(discover_runs(results_dir), data_dir)


def run_query(model_dir: Path, sql: str, data_dir: Optional[Path] = None) -> Any:
    """Result of one SQL statement on one run, as a pandas DataFrame."""
    return connect(model_dir, data_dir).execute(sql).df()


def run_query_all(results_dir: Path, sql: str, data_dir: Optional[Path] = None) -> Any:
    """Result of one SQL statement on every run, as a pandas DataFrame."""
    return connect_all(results_dir, data_dir).execute(sql).df()


def rq1_times(
    con: "duckdb.DuckDBPyConnection",
    goal: str,
    replayable: str = "all",
    weight: str = "instance",
    log: Optional[str] = None,
) -> Any:
    """Time statistics (solved only) per bucket, as a DataFrame.

    replayable: "all" (no filter), "only" (every case of the instance is
        replayable) or "exclude" (no case is).
    weight: "instance" (one OPTIC run counts once) or "case" (once per case).
    log: restrict to one log (needed when the connection holds several).
    """
    if replayable not in ("all", "only", "exclude"):
        raise ValueError(f"replayable must be all/only/exclude, got {replayable!r}")
    if weight not in ("instance", "case"):
        raise ValueError(f"weight must be instance/case, got {weight!r}")
    source = "instances_view" if weight == "instance" else "cases"
    if weight == "instance":
        flt = {"all": "true", "only": "replayability = 'all'", "exclude": "replayability = 'none'"}[replayable]
    else:
        flt = {"all": "true", "only": "coalesce(is_replayable, false)",
               "exclude": "NOT coalesce(is_replayable, false)"}[replayable]
    stats = ", ".join(
        f"avg({f}) AS {f}_mean, var_samp({f}) AS {f}_var, median({f}) AS {f}_median" for f in TIME_FIELDS
    )
    params: List[Any] = [goal]
    log_filter = ""
    if log is not None:
        log_filter, params = " AND log = ?", [goal, log]
    return con.execute(
        f"SELECT bucket, count(*) AS n, {stats} FROM {source} "
        f"WHERE goal = ? AND status = 'solved' AND {flt}{log_filter} GROUP BY bucket ORDER BY min(k)",
        params,
    ).df()
