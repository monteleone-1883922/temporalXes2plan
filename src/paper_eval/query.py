"""Query every result of the paper evaluation with SQL (DuckDB, in memory, no server).

The files of the run stay the only source of truth; nothing is written. A
model run is identified by (log, split_hash), and every table carries those two
columns, so several logs can live in the same database (`connect_all`).

Tables (T = built from a small JSON document in Python, J = DuckDB reads the
JSONL file in place):

  runs                  one row per (log, split_hash)                 T split.json, prefixes_meta.json
  logs                  one row per log                               T <data_dir>/<log>.stats.json
  case_split            one row per case: part = train_fit/val/test   T split.json
  models                one row per run: net, attributes, PDDL, ...   T model_stats.json (flattened)
  search_trials         one row per TPE trial                         T search_trials.json
  attribute_catalog     one row per attribute and its fate            T current.json + attribute_filter.json
  transition_durations  one row per transition with a duration        T current.json
  prefixes              one row per (test case, k, goal)              J prefixes.jsonl
  instances             one row per deduplicated instance             J instances.jsonl
  plan_meta             one row per run: OPTIC settings, environment  T plan_meta.json (flattened)
  plan_results          one row per instance: OPTIC outcome and times J plan_results.jsonl

Views:

  cases           prefixes LEFT JOIN plan_results on (log, split_hash, group_id):
                  one row per (case, k, goal) with the outcome of its instance
                  (status NULL if not run), plus fits_deadline_best_case and
                  fits_deadline_worst_case (deadline goal only, see below).
  instances_view  one row per instance: n_cases, n_cases_replayable,
                  replayability ('all' / 'mixed' / 'none') plus the plan columns.

Keys: runs(log, split_hash) is the hub. prefixes, instances and plan_results
join on group_id inside a run; prefixes.case_id points to case_split.

Weighting. Several cases share one instance and so one OPTIC run. Averaging a
time over `cases` counts that run once per case; averaging over `instances_view`
counts it once. `rq1_times` lets you pick (weight="instance" or "case").

fits_deadline_*: for a deadline record, whether the model's replay can reach
the end within the record's deadline: reached_end AND min (best case) or max
(worst case) modeled duration of the tail <= deadline_s. OPTIC chooses each
duration inside [min, max], so the best case is the one that decides feasibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb
import pandas as pd

PLAN_COLUMNS = (
    "status", "grounding_s", "planning_s", "first_plan_s", "wall_s", "cpu_s",
    "states_evaluated", "makespan_s", "plan_length", "timeout_s", "memory_mb", "message",
)
TIME_FIELDS = ("grounding_s", "planning_s", "wall_s")
_PLAN_TEXT_COLUMNS = {"status", "message"}

# Column -> pandas dtype of the tables built in Python, so an empty table (file
# missing) still has the right types.
_SCHEMAS: Dict[str, Dict[str, str]] = {
    "runs": {
        "log": "object", "split_hash": "object", "model_dir": "object", "log_sha256": "object",
        "test_pct": "float64", "val_pct": "float64", "n_train_fit": "Int64", "n_val": "Int64",
        "n_test": "Int64", "deadline_variant_only": "boolean",
    },
    "case_split": {"log": "object", "split_hash": "object", "case_id": "object", "part": "object"},
    "search_trials": {
        "log": "object", "split_hash": "object", "trial_number": "Int64", "score": "float64",
        "rejected_reason": "object", "max_effect_groups": "Int64",
    },
    "attribute_catalog": {
        "log": "object", "split_hash": "object", "attribute": "object", "type": "object",
        "n_values": "Int64", "n_bin_boundaries": "Int64", "status": "object", "reason": "object",
        "kept_by_user": "boolean",
    },
    "transition_durations": {
        "log": "object", "split_hash": "object", "transition": "object", "activity_name": "object",
        "effective_min": "float64", "effective_max": "float64", "mean": "float64",
        "std_dev": "float64", "observed_min": "float64", "observed_max": "float64",
        "count": "Int64", "source": "object",
    },
    "models": {"log": "object", "split_hash": "object"},
    "plan_meta": {"log": "object", "split_hash": "object"},
    "logs": {"log": "object"},
}


@dataclass(frozen=True)
class Run:
    """One model run: results/<log>/<split_hash>/."""

    log: str
    split_hash: str
    model_dir: Path

    @property
    def log_dir(self) -> Path:
        return self.model_dir.parent


# ---------------------------------------------------------------------------
# Reading the small JSON documents
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _flatten(data: Dict[str, Any], skip: Iterable[str] = (), prefix: str = "") -> Dict[str, Any]:
    """Nested dicts become one level: {"a": {"b": 1}} -> {"a_b": 1}. Lists are
    kept as JSON text, so a row never holds a Python list."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if not prefix and key in skip:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, prefix=f"{name}_"))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value)
        else:
            out[name] = value
    return out


def _frame(name: str, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """DataFrame of `rows` typed by _SCHEMAS[name]. Columns the schema does not
    list (the flattened tables) are kept as they are."""
    schema = _SCHEMAS[name]
    columns = list(schema) + sorted({c for r in rows for c in r} - set(schema))
    df = pd.DataFrame(rows, columns=columns)
    for column, dtype in schema.items():
        df[column] = df[column].astype(dtype)
    return df


def _runs_table(runs: List[Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        split = _read_json(run.log_dir / "split.json") or {}
        params = split.get("params") or {}
        meta = _read_json(run.model_dir / "prefixes_meta.json") or {}
        rows.append({
            "log": run.log, "split_hash": run.split_hash, "model_dir": str(run.model_dir),
            "log_sha256": split.get("log_sha256"),
            "test_pct": params.get("test_pct"), "val_pct": params.get("val_pct"),
            "n_train_fit": len(split["train_fit_ids"]) if split else None,
            "n_val": len(split["val_ids"]) if split else None,
            "n_test": len(split["test_ids"]) if split else None,
            # A run made before the option existed used the min rule.
            "deadline_variant_only": bool(meta.get("deadline_variant_only", False)),
        })
    return _frame("runs", rows)


def _case_split_table(runs: List[Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        split = _read_json(run.log_dir / "split.json")
        if not split:
            continue
        for part, key in (("train_fit", "train_fit_ids"), ("val", "val_ids"), ("test", "test_ids")):
            rows += [{"log": run.log, "split_hash": run.split_hash, "case_id": str(c), "part": part}
                     for c in split.get(key, [])]
    return _frame("case_split", rows)


def _search_trials_table(runs: List[Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for trial in _read_json(run.model_dir / "search_trials.json") or []:
            rows.append({"log": run.log, "split_hash": run.split_hash, **trial})
    return _frame("search_trials", rows)


def _attribute_catalog_table(runs: List[Run]) -> pd.DataFrame:
    """The final catalog (current.json) plus the attributes the filters removed
    before it (attribute_filter.json), each with its status:
    kept / removed_noise / removed_unused / ignored_by_user."""
    rows = []
    for run in runs:
        current = _read_json(run.model_dir / "current.json") or {}
        report = _read_json(run.model_dir / "attribute_filter.json") or {}
        kept_by_user = set(report.get("kept_by_user", []))
        base = {"log": run.log, "split_hash": run.split_hash}
        seen = set()
        for name, entry in (current.get("attribute_catalog") or {}).items():
            seen.add(name)
            rows.append({
                **base, "attribute": name, "type": entry.get("type"),
                "n_values": len(entry.get("possible_values") or []),
                "n_bin_boundaries": len(entry.get("bin_boundaries") or []),
                "status": "kept", "reason": None, "kept_by_user": name in kept_by_user,
            })
        removed = []  # (attribute, status, reason)
        noise = report.get("noise_filter") or {}
        if isinstance(noise, dict):
            removed += [(a, "removed_noise", str(why)) for a, why in noise.items()]
        else:
            removed += [(a, "removed_noise", None) for a in noise]
        removed += [(a, "removed_unused", None) for a in report.get("unused_attribute_filter", [])]
        removed += [(a, "ignored_by_user", None) for a in report.get("ignored_by_user", [])]
        for name, status, reason in removed:
            if name in seen:
                continue
            seen.add(name)
            rows.append({
                **base, "attribute": name, "type": None, "n_values": None, "n_bin_boundaries": None,
                "status": status, "reason": reason, "kept_by_user": name in kept_by_user,
            })
    return _frame("attribute_catalog", rows)


def _transition_durations_table(runs: List[Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        current = _read_json(run.model_dir / "current.json") or {}
        for name, transition in (current.get("transitions") or {}).items():
            duration = transition.get("duration") or {}  # None: transition without a duration
            rows.append({
                "log": run.log, "split_hash": run.split_hash, "transition": name,
                "activity_name": transition.get("activity_name"),
                **{k: duration.get(k) for k in (
                    "effective_min", "effective_max", "mean", "std_dev",
                    "observed_min", "observed_max", "count", "source")},
            })
    return _frame("transition_durations", rows)


def _flattened_table(name: str, runs: List[Run], filename: str, skip: Iterable[str] = ()) -> pd.DataFrame:
    """One row per run from a single-object JSON document, flattened."""
    rows = []
    for run in runs:
        document = _read_json(run.model_dir / filename)
        if document:
            rows.append({"log": run.log, "split_hash": run.split_hash, **_flatten(document, skip)})
    return _frame(name, rows)


def _logs_table(runs: List[Run], data_dir: Optional[Path]) -> pd.DataFrame:
    rows, done = [], set()
    for run in runs:
        if run.log in done or data_dir is None:
            continue
        done.add(run.log)
        document = _read_json(Path(data_dir) / f"{run.log}.stats.json")
        if document:
            rows.append({"log": run.log, **_flatten(document)})
    return _frame("logs", rows)


# ---------------------------------------------------------------------------
# The JSONL files: DuckDB reads them in place
# ---------------------------------------------------------------------------

def _sql_list(paths: List[Path]) -> str:
    return "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def _jsonl_view(
    con: "duckdb.DuckDBPyConnection", name: str, runs: List[Run], filename: str,
    drop: Iterable[str] = (), tolerate_partial_line: bool = False,
) -> bool:
    """View `name` over `filename` of every run that has it, with the columns
    log and split_hash in front. Returns False (no view) when no run has the file."""
    files = [(r, r.model_dir / filename) for r in runs]
    files = [(r, f) for r, f in files if f.is_file() and f.stat().st_size > 0]
    if not files:
        return False
    owners = pd.DataFrame(
        [{"file": str(f), "log": r.log, "split_hash": r.split_hash} for r, f in files]
    )
    con.register(f"_owners_{name}", owners)
    options = "union_by_name=true, filename=true, sample_size=-1"
    if tolerate_partial_line:
        # plan_results.jsonl may end in a half-written line while the plan phase runs.
        options += ", ignore_errors=true"
    source = f"read_json_auto({_sql_list([f for _, f in files])}, {options})"
    # Our own log/split_hash come first; a column of the file with the same name
    # (plan_results.jsonl has "log") and the join helpers are left out. The list
    # is built here instead of with SQL lambdas, whose syntax changes between versions.
    hidden = {"filename", "file", "log", "split_hash", *drop}
    names = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
    columns = ", ".join('j."' + c.replace('"', '""') + '"' for c in names if c not in hidden)
    con.execute(
        f"CREATE VIEW {name} AS SELECT o.log, o.split_hash, {columns} "
        f"FROM {source} j JOIN _owners_{name} o ON j.filename = o.file"
        # ignore_errors turns a broken line into an all-NULL row; every real row has a group_id.
        + (" WHERE j.group_id IS NOT NULL" if tolerate_partial_line else "")
    )
    return True


def _empty_plan_results(con: "duckdb.DuckDBPyConnection") -> None:
    """Same columns as plan_results.jsonl's measures, no rows."""
    cols = ", ".join(
        f"NULL::{'VARCHAR' if c in _PLAN_TEXT_COLUMNS else 'DOUBLE'} AS {c}" for c in PLAN_COLUMNS
    )
    con.execute(
        "CREATE VIEW plan_results AS SELECT NULL::VARCHAR AS log, NULL::VARCHAR AS split_hash, "
        f"NULL::BIGINT AS group_id, {cols} WHERE false"
    )


def _empty_instances(con: "duckdb.DuckDBPyConnection") -> None:
    con.execute(
        "CREATE VIEW instances AS SELECT NULL::VARCHAR AS log, NULL::VARCHAR AS split_hash, "
        "NULL::BIGINT AS group_id, NULL::VARCHAR AS goal, NULL::BIGINT AS k, "
        "NULL::VARCHAR AS bucket WHERE false"
    )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def _build(
    runs: List[Run], data_dir: Optional[Path] = None,
) -> "duckdb.DuckDBPyConnection":
    if not _has_prefixes(runs):
        where = runs[0].model_dir if len(runs) == 1 else "any run"
        raise FileNotFoundError(f"{where}/prefixes.jsonl not found: run the 'prefixes' phase first")
    con = duckdb.connect()
    tables = {
        "runs": _runs_table(runs),
        "case_split": _case_split_table(runs),
        "search_trials": _search_trials_table(runs),
        "attribute_catalog": _attribute_catalog_table(runs),
        "transition_durations": _transition_durations_table(runs),
        "models": _flattened_table(
            "models", runs, "model_stats.json", skip=("model_dir", "split", "attribute_filter", "log")),
        "plan_meta": _flattened_table("plan_meta", runs, "plan_meta.json"),
        "logs": _logs_table(runs, data_dir),
    }
    for name, frame in tables.items():
        con.register(name, frame)

    _jsonl_view(con, "prefixes", runs, "prefixes.jsonl")
    if not _jsonl_view(con, "instances", runs, "instances.jsonl"):
        _empty_instances(con)
    if not _jsonl_view(con, "plan_results", runs, "plan_results.jsonl",
                       drop=("log",), tolerate_partial_line=True):
        _empty_plan_results(con)

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


def _has_prefixes(runs: List[Run]) -> bool:
    return any((r.model_dir / "prefixes.jsonl").is_file() for r in runs)


def connect(model_dir: Path, data_dir: Optional[Path] = None) -> "duckdb.DuckDBPyConnection":
    """In-memory connection over ONE run (results/<log>/<split_hash>/).
    data_dir (the folder with <log>.stats.json) fills the `logs` table."""
    model_dir = Path(model_dir)
    return _build([Run(model_dir.parent.name, model_dir.name, model_dir)], data_dir)


def discover_runs(results_dir: Path) -> List[Run]:
    """Every results/<log>/<split_hash>/ that holds a built model."""
    runs: List[Run] = []
    for log_dir in sorted(Path(results_dir).iterdir()):
        if not (log_dir / "split.json").is_file():
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
