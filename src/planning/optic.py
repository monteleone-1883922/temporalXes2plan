"""OPTIC temporal planner — importable wrapper."""

import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent.parent

OPTIC_DIR = PROJECT_ROOT / "optic"
OPTIC_SH = OPTIC_DIR / "optic-clp.sh"
OPTIC_BIN = OPTIC_DIR / "release" / "optic" / "optic-clp"


def is_available() -> bool:
    """Return True if the OPTIC binary has been compiled."""
    return OPTIC_BIN.is_file()


def build(log_fn: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Compile OPTIC by running run-cmake-release then build-release.

    Both scripts must be run from OPTIC_DIR.
    Safe to call if already built (cmake is idempotent).

    Args:
        log_fn: Optional callback for progress lines.

    Returns:
        Dict with keys: success (bool), message (str), stdout (str), stderr (str).
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    for script in ("run-cmake-release", "build-release"):
        if not (OPTIC_DIR / script).exists():
            msg = f"OPTIC build script not found: {OPTIC_DIR / script}"
            _log(msg)
            return {"success": False, "message": msg, "stdout": "", "stderr": ""}

    _log("Building OPTIC — this may take several minutes...")
    all_stdout: List[str] = []

    # Ensure the conda env lib dir is on LIBRARY_PATH so the linker can find
    # libz.so when the system only ships libz.so.1 (no -dev package installed).
    build_env = os.environ.copy()
    conda_lib = Path(sys.executable).resolve().parent.parent / "lib"
    if conda_lib.is_dir():
        existing = build_env.get("LIBRARY_PATH", "")
        build_env["LIBRARY_PATH"] = (
            f"{conda_lib}:{existing}" if existing else str(conda_lib)
        )

    for step, script in enumerate(("run-cmake-release", "build-release"), start=1):
        _log(f"Step {step}/2: {script}")
        try:
            proc = subprocess.Popen(
                ["bash", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(OPTIC_DIR),
                env=build_env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                all_stdout.append(line)
                _log(line)
            proc.wait()
        except Exception as exc:
            msg = f"Build execution error at step {step}: {exc}"
            _log(msg)
            return {"success": False, "message": msg, "stdout": "\n".join(all_stdout), "stderr": str(exc)}

        if proc.returncode != 0:
            msg = f"OPTIC build failed at step {step} ({script}), exit {proc.returncode}."
            _log(msg)
            return {"success": False, "message": msg, "stdout": "\n".join(all_stdout), "stderr": ""}

    _log("OPTIC build complete.")
    return {"success": True, "message": "Build complete.", "stdout": "\n".join(all_stdout), "stderr": ""}


def run(
    pddl_dir: Path,
    stop_at_first: bool = True,
    ignore_costs: bool = False,
    timeout: int = 60,
    memory_mb: int = 4000,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Execute OPTIC on domain.pddl + problem.pddl inside pddl_dir.

    OPTIC writes the plan to stdout in temporal format:
        <timestamp>: (<action>) [<duration>]

    Args:
        pddl_dir: Directory containing domain.pddl and problem.pddl.
        stop_at_first: Pass -N flag (stop after first solution, no cost opt.).
        ignore_costs: Pass -c flag (treat all actions as unit cost).
        timeout: Hard timeout in seconds (enforced via subprocess).
        memory_mb: Soft memory limit in MB (enforced via ulimit -v in bash wrapper).
        log_fn: Optional progress callback.

    Returns:
        Dict with keys: success, solvability, plan_text, plan_actions,
        metrics, stdout, stderr, message.
    """
    _empty = _empty_result()

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    if not is_available():
        _log("OPTIC not built — starting build...")
        result = build(log_fn)
        if not result["success"]:
            return {**_empty, "message": result["message"]}

    if not is_available():
        return {**_empty, "message": "OPTIC binary not found after build attempt."}

    domain = (pddl_dir / "domain.pddl").resolve()
    problem = (pddl_dir / "problem.pddl").resolve()

    if not domain.exists() or not problem.exists():
        return {**_empty, "message": "domain.pddl or problem.pddl not found in pddl directory."}

    flags: List[str] = []
    if stop_at_first:
        flags.append("-N")
    if ignore_costs:
        flags.append("-c")
    memory_kb = memory_mb * 1024

    # Wrap in bash so ulimit applies; run from OPTIC_DIR so the .sh script
    # resolves ./release/optic/optic-clp correctly.
    inner_cmd = " ".join(
        [f"ulimit -v {memory_kb};",
         str(OPTIC_SH)] + flags + [str(domain), str(problem)]
    )
    cmd = ["bash", "-c", inner_cmd]

    mode = ("stop at first" if stop_at_first else "optimise") + (", ignore costs" if ignore_costs else "")
    _log(f"Mode: {mode}")
    _log(f"Timeout: {timeout}s  Memory: {memory_mb}MB")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(OPTIC_DIR),
            # New session (own process group) so a timeout can kill the
            # whole bash -> optic-clp.sh -> optic-clp chain via killpg
            # below, not just the outer `bash -c` process — cmd's shebang-
            # less optic-clp.sh (see OPTIC_SH) is itself interpreted by a
            # nested shell that forks the actual optic-clp binary, so
            # subprocess.run(timeout=...)'s default .kill() (SIGKILL to the
            # immediate child only) leaves that binary orphaned and still
            # running/consuming memory in the background after we've
            # already reported "timeout" and moved on to the next query.
            start_new_session=True,
        )
    except Exception as exc:
        return {**_empty, "solvability": "error",
                "stderr": str(exc), "message": f"Execution error: {exc}"}

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _log("Timeout expired.")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # already exited between the timeout and the killpg call
        proc.communicate()  # reap the process, discard partial output
        return {**_empty,
                "solvability": "timeout",
                "message": f"Planner timed out after {timeout}s"}

    returncode = proc.returncode
    _log(f"Planner finished (exit code {returncode})")

    plan_lines, plan_actions, makespan = _parse_optic_stdout(out)
    plan_text: Optional[str] = None

    if plan_actions:
        plan_text = "\n".join(plan_lines)
        (pddl_dir / "plan.txt").write_text(plan_text, encoding="utf-8")
        _log(f"Plan saved — {len(plan_actions)} action(s)")

    metrics = _parse_optic_metrics(out, err, makespan)
    metrics["solution_length"] = len(plan_actions) if plan_actions else None

    solvability = _classify(returncode, plan_text is not None, out, err)
    success = solvability == "solved"
    message = (
        f"Plan found — {len(plan_actions)} action(s)" if success
        else "Problem proved unsolvable" if solvability == "unsolvable_structural"
        else f"Planner ran out of memory (exit {returncode})" if solvability == "out_of_memory"
        else f"No solution found (exit {returncode})"
    )
    _log(message)

    return {
        "success": success,
        "solvability": solvability,
        "plan_text": plan_text,
        "plan_actions": plan_actions,
        "metrics": metrics,
        "stdout": out,
        "stderr": err,
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
        "metrics": {k: None for k in ("expanded_nodes", "search_time", "solution_length", "total_time", "cost", "duration")},
        "stdout": "",
        "stderr": "",
        "message": "",
    }


def _parse_optic_stdout(stdout: str):
    """Extract the last (best) plan from OPTIC stdout.

    OPTIC may output multiple improving solutions; each block starts with
    ';;; Solution Found'. We take the last one.

    Returns:
        Tuple of (raw_plan_lines, action_name_list, makespan). `makespan`
        is the plan's simulated completion time — max(timestamp + duration)
        over all timed lines in the block, not just the last line, since a
        temporal plan can have concurrent actions finishing out of order.
    """
    # Split on solution delimiters, keep the last block
    blocks = re.split(r";;; Solution Found", stdout)
    if len(blocks) < 2:
        return [], [], None

    last_block = blocks[-1]
    plan_lines = []
    actions = []
    makespan: Optional[float] = None

    for line in last_block.splitlines():
        # Temporal plan line: "  0.000: (action_name) [duration]"
        m = re.match(r"^\s*([\d.]+)\s*:\s*\(([^)]+)\)\s*\[([\d.]+)\]", line)
        if m:
            plan_lines.append(line.strip())
            name = _normalise(m.group(2).split()[0])
            if name and not name.startswith("tau_"):
                actions.append(name)
            end_time = float(m.group(1)) + float(m.group(3))
            makespan = end_time if makespan is None else max(makespan, end_time)

    return plan_lines, actions, makespan


def _parse_optic_metrics(stdout: str, stderr: str, makespan: Optional[float] = None) -> Dict[str, Any]:
    combined = stdout + "\n" + stderr
    metrics: Dict[str, Any] = {k: None for k in ("expanded_nodes", "search_time", "solution_length", "total_time", "cost", "duration")}

    m = re.search(r"States evaluated:\s*(\d+)", combined)
    if m:
        metrics["expanded_nodes"] = int(m.group(1))

    m = re.search(r"Time\s+([\d.]+)", combined)
    if m:
        metrics["total_time"] = float(m.group(1))
        metrics["search_time"] = metrics["total_time"]

    m = re.search(r";\s*Cost:\s*([\d.]+)", combined)
    if m:
        metrics["cost"] = float(m.group(1))

    metrics["duration"] = makespan

    return metrics


# OPTIC is invoked under `ulimit -v {memory_kb}` (see run()) — when it
# exceeds that cap, malloc/new fails and (since OPTIC doesn't catch the
# allocation failure) the process terminates via std::terminate() -> abort(),
# i.e. SIGABRT, observed as return_code 134 (128 + signal 6) with
# "terminate called after throwing an instance of 'std::bad_alloc'" on
# stderr — confirmed directly from a real failure
# (results/55/failures/55_S74983_Q3.json). Detected here so the evaluation
# summary can report out-of-memory failures separately from other
# resource-limit ones instead of lumping everything into
# "unsolvable_resource".
_OPTIC_OOM_RETURN_CODE = 134


def _classify(return_code: int, plan_exists: bool, stdout: str, stderr: str) -> str:
    combined = (stdout + "\n" + stderr).lower()
    if plan_exists:
        return "solved"
    if "unsolvable" in combined or "no solution" in combined:
        return "unsolvable_structural"
    if any(kw in combined for kw in (
        "parse error", "error in domain", "error in problem",
        "unrecognised", "undefined type", "undefined predicate",
        "could not evaluate", "type error",
    )):
        return "unsolvable_parse"
    if return_code == _OPTIC_OOM_RETURN_CODE or any(
        kw in combined for kw in ("bad_alloc", "cannot allocate memory", "out of memory")
    ):
        return "out_of_memory"
    if return_code != 0:
        return "unsolvable_resource"
    return "unsolvable_resource"


def _normalise(name: str) -> str:
    if name.startswith("execute_"):
        name = name[len("execute_"):]
    name = name.split("_DETDUP")[0]
    name = re.sub(r"_v\d+$", "", name)
    return name
