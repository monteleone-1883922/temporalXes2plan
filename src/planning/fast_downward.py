"""Fast Downward planner — importable wrapper with path-aware execution."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent.parent

FD_SCRIPT  = PROJECT_ROOT / "vendor" / "downward" / "fast-downward.py"
BUILDS_DIR = PROJECT_ROOT / "vendor" / "downward" / "builds" / "release"
SETUP_SH   = PROJECT_ROOT / "scripts" / "setup.sh"

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
    """Return True if Fast Downward has been compiled.

    Checks for the actual search binary, not just the builds/release
    directory -- an interrupted or partial CMake build can leave that
    directory in place (CMakeFiles/ etc.) without ever producing bin/downward,
    which would otherwise be silently mistaken for a completed build.
    """
    return (BUILDS_DIR / "bin" / "downward").is_file()


def get_search_configs() -> Dict[str, Any]:
    """Return serializable search config catalog with group labels."""
    key_to_group = {k: g for g, keys in _GROUPS.items() for k in keys}
    return {
        key: {"type": cfg["type"], "config": cfg["config"], "group": key_to_group[key]}
        for key, cfg in _SEARCH_CONFIGS.items()
    }


def build(log_fn: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Compile Fast Downward by running scripts/setup.sh.

    Handles git submodule initialisation and CMake build.
    Safe to call if already built (setup.sh is idempotent).

    Args:
        log_fn: Optional callback for progress lines.

    Returns:
        Dict with keys: success (bool), message (str), stdout (str), stderr (str).
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    if not SETUP_SH.exists():
        msg = f"Setup script not found: {SETUP_SH}"
        _log(msg)
        return {"success": False, "message": msg, "stdout": "", "stderr": ""}

    _log("Building Fast Downward — this may take several minutes...")
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    try:
        proc = subprocess.Popen(
            ["bash", str(SETUP_SH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            stdout_lines.append(line)
            _log(line)
        proc.wait()
    except Exception as exc:
        msg = f"Build execution error: {exc}"
        _log(msg)
        return {"success": False, "message": msg, "stdout": "", "stderr": str(exc)}

    stdout = "\n".join(stdout_lines)
    if proc.returncode == 0:
        _log("Fast Downward build complete.")
        return {"success": True, "message": "Build complete.", "stdout": stdout, "stderr": ""}
    else:
        msg = f"Build failed (exit {proc.returncode})."
        _log(msg)
        return {"success": False, "message": msg, "stdout": stdout, "stderr": ""}


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

    if not is_built():
        _log("Fast Downward not built — starting build...")
        result = build(log_fn)
        if not result["success"]:
            return {**_empty, "message": result["message"]}

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
                "solvability": "timeout",
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
        # read_text() always returns a str (never None); the read result is
        # kept in its own str-typed local so downstream calls in this block
        # don't need to re-narrow the Optional[str] `plan_text` declared above.
        plan_text_content: str = plan_files[0].read_text(encoding="utf-8")
        plan_text = plan_text_content
        (pddl_dir / "plan.txt").write_text(plan_text_content, encoding="utf-8")
        plan_actions = _parse_classical_plan(plan_text_content)
        _log(f"Plan saved — {len(plan_actions)} action(s)")
        _cleanup(pddl_dir, keep={"domain.pddl", "problem.pddl", "plan.txt"})

    solvability = _classify(proc.returncode, plan_text is not None, proc.stdout, proc.stderr)
    success = solvability == "solved"
    message = (
        f"Plan found — {len(plan_actions)} action(s)" if success
        else "Problem proved unsolvable" if solvability == "unsolvable_structural"
        else f"Planner ran out of memory (exit {proc.returncode})" if solvability == "out_of_memory"
        else f"Planner ran out of time (exit {proc.returncode})" if solvability == "timeout"
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


# Fast Downward's own documented exit codes (vendor/downward/driver/returncodes.py)
# that distinguish an out-of-memory give-up from an out-of-time give-up —
# used so the evaluation summary can report the two separately instead of
# lumping every resource-limit failure into "unsolvable_resource".
_FD_OUT_OF_MEMORY_CODES = {20, 22, 24}  # TRANSLATE/SEARCH_OUT_OF_MEMORY(_AND_TIME)
_FD_OUT_OF_TIME_CODES = {21, 23}        # TRANSLATE/SEARCH_OUT_OF_TIME


def _classify(return_code: int, plan_exists: bool, stdout: str, stderr: str) -> str:
    combined = (stdout + "\n" + stderr).lower()
    if return_code == 0 and plan_exists:
        return "solved"
    if return_code in (10, 11) or "completely explored state space" in combined:
        return "unsolvable_structural"
    if return_code in _FD_OUT_OF_MEMORY_CODES or any(
        kw in combined for kw in ("out of memory", "memory exhausted")
    ):
        return "out_of_memory"
    if return_code in _FD_OUT_OF_TIME_CODES or any(
        kw in combined for kw in ("out of time", "time limit")
    ):
        return "timeout"
    return "unsolvable_resource"


def parse_plan_text(plan_text: str) -> List[str]:
    """Parse a plan string and return normalised, tau-filtered activity names.

    Handles both classical FD format ``(action_name)`` and temporal OPTIC
    format ``0.000: (action_name) [duration]``.

    Args:
        plan_text: Raw plan file content.

    Returns:
        List of sanitised activity name strings with exec_ prefix and
        variant/dedup suffixes stripped, tau_ actions excluded.
    """
    actions = []
    for line in plan_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = re.search(r"\(([^)]+)\)", line)
        if not m:
            continue
        name = _normalise(m.group(1).split()[0])
        if name and not name.startswith("tau_"):
            actions.append(name)
    return actions


def _parse_classical_plan(plan_text: str) -> List[str]:
    """Parse FD sas_plan format: each non-comment line is (action_name ...)."""
    return parse_plan_text(plan_text)


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
