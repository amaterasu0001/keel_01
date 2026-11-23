"""Signals for patches written by external agents (no Keel run, so no logprobs or adversarial tests).

Everything here needs only the issue text, the diff and, if the submission ships one, the agent's
trajectory. The LLM-judge signal is optional and calls any Keel generator.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

from ..diffutil import diff_stats

EXTERNAL_FEATURES = [
    "log_diff_size", "n_files", "n_hunks", "touches_tests",
    "issue_token_overlap", "issue_file_mention", "issue_identifier_hits",
    "has_traj", "traj_log_steps", "traj_test_runs", "traj_log_files_viewed", "traj_edit_steps",
    "has_judge", "judge_correct_prob",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PATHLIKE = re.compile(r"[\w./-]+\.(?:py|pyx|js|ts|java|c|cc|cpp|h|rb|go|rs)\b")
_STOP = {"the", "and", "for", "with", "this", "that", "from", "not", "but", "are", "was", "has", "have",
         "def", "self", "return", "none", "true", "false", "import", "class", "when", "should", "would",
         "can", "will", "use", "using", "get", "set", "into", "its", "also", "there", "what", "which"}


def _tokens(text: str) -> set[str]:
    out = set()
    for t in _IDENT.findall(text):
        t = t.lower()
        if t not in _STOP:
            out.add(t)
            out.update(p for p in t.split("_") if len(p) > 2 and p not in _STOP)
    return out


def _changed_text(diff: str) -> str:
    lines = []
    for ln in diff.splitlines():
        if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")):
            lines.append(ln[1:])
        elif ln.startswith("@@"):
            lines.append(ln.split("@@")[-1])  # the enclosing function name, if git printed it
    return "\n".join(lines)


def issue_features(problem_statement: str, diff: str) -> dict[str, float]:
    """How well the change lines up with what the issue talks about."""
    issue_toks = _tokens(problem_statement or "")
    diff_toks = _tokens(_changed_text(diff))
    st = diff_stats(diff)
    overlap = len(issue_toks & diff_toks) / len(diff_toks) if diff_toks else 0.0
    mentioned = {Path(p).name for p in _PATHLIKE.findall(problem_statement or "")}
    touched = {Path(p).name for p in st.paths}
    file_mention = (len(mentioned & touched) / len(touched)) if touched and mentioned else 0.0
    hits = len(issue_toks & diff_toks)
    return {"issue_token_overlap": overlap, "issue_file_mention": file_mention,
            "issue_identifier_hits": math.log1p(hits)}


def diff_features(diff: str) -> dict[str, float]:
    from ..sandbox import touches_protected
    from ..types import DEFAULT_PROTECTED
    st = diff_stats(diff)
    return {"log_diff_size": math.log1p(st.added + st.removed), "n_files": float(st.files),
            "n_hunks": float(st.hunks), "touches_tests": float(touches_protected(diff, DEFAULT_PROTECTED))}


_TEST_CMD = re.compile(r"\b(pytest|py\.test|tox|unittest|nosetests|manage\.py test|runtests)\b")
_VIEW_CMD = re.compile(r"^\s*(open|cat|less|head|tail|sed -n|view|str_replace_editor view|find_file|goto)\b")
_EDIT_CMD = re.compile(r"^\s*(edit|str_replace_editor (?:str_replace|create|insert)|create|insert|apply_patch)\b")


def _parse_maybe(x):
    """JSON, or a Python-repr string of a list/dict (OpenHands exports both), or the value itself."""
    if not isinstance(x, str):
        return x
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(x)
        except (ValueError, SyntaxError, TypeError):
            continue
    return x


def _tool_call_actions(calls) -> list[str]:
    """OpenAI-style tool calls -> 'execute_bash pytest -q', 'str_replace_editor view path', ..."""
    out = []
    for call in _parse_maybe(calls) or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name") or ""
        args = _parse_maybe(fn.get("arguments") or call.get("arguments") or {})
        if isinstance(args, dict):
            if name in ("execute_bash", "bash", "run", "shell"):
                out.append(str(args.get("command", "")))
            else:
                out.append(" ".join(str(x) for x in (name, args.get("command", ""), args.get("path", "")) if x))
        else:
            out.append(f"{name} {args}")
    return out


def _actions(traj) -> list[str]:
    """Best effort over common formats (SWE-agent .traj, OpenHands / mini-SWE-agent histories)."""
    if isinstance(traj, dict):
        for key in ("trajectory", "history", "messages", "steps"):
            if isinstance(traj.get(key), list):
                traj = traj[key]
                break
        else:
            return []
    if not isinstance(traj, list):
        return []
    out = []
    for step in traj:
        if isinstance(step, dict):
            if step.get("tool_calls"):
                out += _tool_call_actions(step["tool_calls"])
                continue
            a = step.get("action") or step.get("command") or ""
            if not a and step.get("role") == "assistant":
                content = _parse_maybe(step.get("content") or "")
                if isinstance(content, list):  # [{"type": "text", "text": ...}]
                    content = " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
                a = content
            if isinstance(a, dict):
                a = json.dumps(a)
            if isinstance(a, list):
                a = " ".join(str(x) for x in a)
            if a and str(a).strip():
                out.append(str(a))
        elif isinstance(step, str):
            out.append(step)
    return out


def trajectory_features(traj) -> dict[str, float]:
    acts = _actions(traj) if traj is not None else []
    if not acts:
        return {"has_traj": 0.0, "traj_log_steps": 0.0, "traj_test_runs": 0.0, "traj_log_files_viewed": 0.0,
                "traj_edit_steps": 0.0}
    files = set()
    for a in acts:
        if _VIEW_CMD.search(a):
            files.update(_PATHLIKE.findall(a))
    return {"has_traj": 1.0, "traj_log_steps": math.log1p(len(acts)),
            "traj_test_runs": float(sum(1 for a in acts if _TEST_CMD.search(a))),
            "traj_log_files_viewed": math.log1p(len(files)),
            "traj_edit_steps": float(sum(1 for a in acts if _EDIT_CMD.search(a)))}


def load_trajectory(traj_dir: Path, instance_id: str):
    """Find and parse an instance's trajectory file in a submission's trajs/ folder, or None."""
    if not traj_dir.exists():
        return None
    for p in sorted(traj_dir.glob(f"{instance_id}*")):
        if p.is_dir():
            inner = sorted(p.glob("*.traj")) + sorted(p.glob("*.json"))
            if not inner:
                continue
            p = inner[0]
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            return json.loads(text)
        except (json.JSONDecodeError, OSError):
            try:  # JSON lines
                return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            except Exception:  # noqa: BLE001
                return None
    return None


JUDGE_SYSTEM = ("You review a code patch written to resolve a GitHub issue. Decide whether it correctly and "
                "completely resolves the issue without breaking other behaviour. Reply with one line: "
                "PROBABILITY: <a number between 0 and 1 that the patch is correct>.")
_PROB = re.compile(r"PROBABILITY:\s*([01](?:\.\d+)?)", re.I)


def judge_features(generator, problem_statement: str, diff: str, max_chars: int = 12000) -> dict[str, float]:
    """LLM-as-judge probability of correctness. `generator` is any Keel generator (or None)."""
    if generator is None:
        return {"has_judge": 0.0, "judge_correct_prob": 0.5}
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"## Issue\n{problem_statement[:max_chars]}\n\n## Patch\n```diff\n"
                                        f"{diff[:max_chars]}\n```"}]
    comps = generator.complete(msgs, n=1, temperature=0.0, max_tokens=64)
    m = _PROB.search(comps[0].text if comps else "")
    if not m:
        return {"has_judge": 0.0, "judge_correct_prob": 0.5}
    return {"has_judge": 1.0, "judge_correct_prob": min(1.0, max(0.0, float(m.group(1))))}
