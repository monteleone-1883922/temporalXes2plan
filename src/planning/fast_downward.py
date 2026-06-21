"""Fast Downward planner — importable wrapper with path-aware execution."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent.parent

FD_SCRIPT = PROJECT_ROOT / "vendor" / "downward" / "fast-downward.py"
BUILDS_DIR = PROJECT_ROOT / "vendor" / "downward" / "builds" / "release"

_SEARCH_CONFIGS: Dict[str, Dict[str, str]] = {
    "astar_blind":         {"type": "search", "config": "astar(blind(), cost_type=one)"},
    "astar_hadd":          {"type": "search", "config": "astar(add(), cost_type=one)"},
    "astar_hff":           {"type": "search", "config": "astar(ff(), cost_type=one)"},
    "astar_lmcut":         {"type": "search", "config": "astar(lmcut(), cost_type=one)"},
    "eager_greedy_blind":  {"type": "search", "config": "eager_greedy([blind()], cost_type=one)"},
    "eager_greedy_hadd":   {"type": "search", "config": "eager_greedy([add()], cost_type=one)"},
    "eager_greedy_hff":    {"type": "search", "config": "eager_greedy([ff()], cost_type=one)"},
    "eager_greedy_lmcut":  {"type": "search", "config": "eager_greedy([lmcut()], cost_type=one)"},
    "seq_sat_lama_2011":   {"type": "alias",  "config": "seq-sat-lama-2011"},
    "seq_opt_bjolp":       {"type": "alias",  "config": "seq-opt-bjolp"},
    "lama_first":          {"type": "alias",  "config": "lama-first"},
}

_GROUPS: Dict[str, List[str]] = {
    "Optimal":      ["astar_blind", "astar_hadd", "astar_hff", "astar_lmcut"],
    "Satisficing":  ["eager_greedy_blind", "eager_greedy_hadd", "eager_greedy_hff", "eager_greedy_lmcut"],
    "Aliases":      ["seq_sat_lama_2011", "seq_opt_bjolp", "lama_first"],
}


def is_built() -> bool:
    """Return True if Fast Downward has been compiled (builds/release exists)."""
    return BUILDS_DIR.is_dir()


def get_search_configs() -> Dict[str, Any]:
    """Return serializable search config catalog with group labels."""
    key_to_group = {k: g for g, keys in _GROUPS.items() for k in keys}
    return {
        key: {"type": cfg["type"], "config": cfg["config"], "group": key_to_group[key]}
        for key, cfg in _SEARCH_CONFIGS.items()
    }


def run(
    pddl_dir: Path,
    search_key: str = "astar_lmcut",
    timeout: int = 30,
    memory_mb: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Execute Fast Downward on domain.pddl + problem.pddl inside pddl_dir.

    Args:
        pddl_dir: Directory containing domain.pddl and problem.pddl.
        search_key: Key from get_search_configs().
        timeout: Hard timeout in seconds.
        memory_mb: Optional memory limit in MB (--overall-memory-limit flag).
        log_fn: Optional callback for progress lines.

    Returns:
        Dict with keys: success, solvability, plan_text, plan_actions,
        metrics, stdout, stderr, message.
    """
    _empty = _empty_result()

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    if not FD_SCRIPT.exists():
        return {**_empty, "message": "Fast Downward not found — run scripts/setup.sh first."}
    if search_key not in _SEARCH_CONFIGS:
        return {**_empty, "message": f"Unknown search config: {search_key!r}"}

    cfg = _SEARCH_CONFIGS[search_key]
    cmd = [sys.executable, str(FD_SCRIPT)]
    if memory_mb is not None:
        cmd += ["--overall-memory-limit", f"{memory_mb}M"]
    if cfg["type"] == "alias":
        cmd += ["--alias", cfg["config"], "domain.pddl", "problem.pddl"]
    else:
        cmd += ["domain.pddl", "problem.pddl", "--search", cfg["config"]]

    _log(f"Search algorithm: {search_key}")
    _log(f"Timeout: {timeout}s" + (f"  Memory: {memory_mb}MB" if memory_mb else ""))

    original_cwd = os.getcwd()
    try:
        os.chdir(pddl_dir)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _log("Timeout expired.")
        return {**_empty,
                "solvability": "unsolvable_resource",
                "message": f"Planner timed out after {timeout}s"}
    except Exception as exc:
        return {**_empty, "solvability": "error",
                "stderr": str(exc), "message": f"Execution error: {exc}"}
    finally:
        os.chdir(original_cwd)

    _log(f"Planner finished (exit code {proc.returncode})")

    metrics = _parse_fd_metrics(proc.stdout, proc.stderr)

    # Locate plan file written by FD (sas_plan or sas_plan.N)
    plan_files = sorted(
        f for f in pddl_dir.iterdir()
        if f.name == "sas_plan" or re.match(r"sas_plan\.\d+$", f.name)
    )
    plan_text: Optional[str] = None
    plan_actions: List[str] = []

    if plan_files:
        plan_text = plan_files[0].read_text(encoding="utf-8")
        (pddl_dir / "plan.txt").write_text(plan_text, encoding="utf-8")
        plan_actions = _parse_classical_plan(plan_text)
        _log(f"Plan saved — {len(plan_actions)} action(s)")
        _cleanup(pddl_dir, keep={"domain.pddl", "problem.pddl", "plan.txt"})

    solvability = _classify(proc.returncode, plan_text is not None, proc.stdout, proc.stderr)
    success = solvability == "solved"
    message = (
        f"Plan found — {len(plan_actions)} action(s)" if success
        else "Problem proved unsolvable" if solvability == "unsolvable_structural"
        else f"No solution found (exit {proc.returncode})"
    )
    _log(message)

    return {
        "success": success,
        "solvability": solvability,
        "plan_text": plan_text,
        "plan_actions": plan_actions,
        "metrics": metrics,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_result() -> Dict[str, Any]:
    return {
        "success": False,
        "solvability": "error",
        "plan_text": None,
        "plan_actions": [],
        "metrics": {k: None for k in ("expanded_nodes", "search_time", "solution_length", "total_time")},
        "stdout": "",
        "stderr": "",
        "message": "",
    }


def _parse_fd_metrics(stdout: str, stderr: str) -> Dict[str, Any]:
    combined = stdout + "\n" + stderr
    metrics: Dict[str, Any] = {k: None for k in ("expanded_nodes", "search_time", "solution_length", "total_time")}

    #find how many noded were expanded
    for pat in (r"Expanded (\d+) state", r"Expanded nodes: (\d+)"):
        m = re.search(pat, combined)
        if m:
            metrics["expanded_nodes"] = int(m.group(1))
            break
    #find search time
    for pat in (r"Search time: ([\d.]+)s", r"Actual search time: ([\d.]+)s"):
        m = re.search(pat, combined)
        if m:
            metrics["search_time"] = float(m.group(1))
            break
    #TODO separate length and cost
    for pat in (r"Plan length: (\d+)", r"Plan cost: (\d+)"):
        m = re.search(pat, combined)
        if m:
            metrics["solution_length"] = int(m.group(1))
            break

    metrics["total_time"] = metrics["search_time"]
    return metrics


def _classify(return_code: int, plan_exists: bool, stdout: str, stderr: str) -> str:
    combined = (stdout + "\n" + stderr).lower()
    if return_code == 0 and plan_exists:
        return "solved"
    if return_code in (10, 11) or "completely explored state space" in combined:
        return "unsolvable_structural"
    if return_code in (12, 20, 21, 22, 23, 24):
        return "unsolvable_resource"
    for kw in ("out of memory", "memory exhausted", "out of time", "time limit"):
        if kw in combined:
            return "unsolvable_resource"
    return "unsolvable_resource"


def _parse_classical_plan(plan_text: str) -> List[str]:
    """Parse FD sas_plan format: each non-comment line is (action_name ...)."""
    actions = []
    for line in plan_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = re.match(r"^\(([^)]+)\)", line)
        if not m:
            continue
        name = _normalise(m.group(1).split()[0])
        if name and not name.startswith("tau_"):
            actions.append(name)
    return actions


def _normalise(name: str) -> str:
    if name.startswith("exec_"):
        name = name[5:]
    name = name.split("_DETDUP")[0]
    name = re.sub(r"_v\d+$", "", name)
    return name


def _cleanup(pddl_dir: Path, keep: set) -> None:
    for f in pddl_dir.iterdir():
        if f.is_file() and f.name not in keep:
            try:
                f.unlink()
            except Exception:
                pass
