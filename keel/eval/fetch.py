"""Download what Oracle 1 and Oracle 2 need: SWE-bench rows and published leaderboard submissions.

For each submission this builds, under one root folder,

    <root>/<submission>/
        README.md, metadata.yaml           from github.com/swe-bench/experiments
        results/results.json               {"resolved": [...], ...}: the original-test label (Oracle 1)
        logs/<instance_id>/patch.diff      the evaluated patch
        logs/<instance_id>/report.json     the official harness report
        trajs/<instance_id>.*              agent trajectories (only with trajs=True)

which is the layout both `keel eval submission` and the SWE-ABS harness read (SWE-ABS takes
`--vaild_model_path <root> --vaild_model_name <submission>` and looks for
`logs/<instance_id>/{report.json,patch.diff}`). So the same folder serves Oracle 1 and Oracle 2.

Sources, as in the experiments repository's own analysis/download_logs.py: the public S3 bucket
`swe-bench-submissions` (anonymous, no credentials), or, when metadata.yaml names an `assets.repo`,
a shallow git clone of the submitter's repository. Downloads resume: existing files are skipped.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

S3 = "https://swe-bench-submissions.s3.amazonaws.com"
RAW = "https://raw.githubusercontent.com/swe-bench/experiments/main/evaluation"
LOG_FILES = ("patch.diff", "report.json")
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

Getter = Callable[[str], bytes | None]


def _default_getter(client: httpx.Client, attempts: int = 5) -> Getter:
    """GET with retries: S3 and GitHub drop connections now and then under parallel downloads."""
    def get(url: str) -> bytes | None:
        for i in range(attempts):
            try:
                r = client.get(url, timeout=60, follow_redirects=True)
                if r.status_code == 404:
                    return None
                if r.status_code in (429, 500, 502, 503, 504) and i < attempts - 1:
                    raise httpx.HTTPStatusError("retry", request=r.request, response=r)
                r.raise_for_status()
                return r.content
            except (httpx.TransportError, httpx.HTTPStatusError):
                if i == attempts - 1:
                    raise
                time.sleep(min(30.0, 1.5 * 2 ** i))
        return None
    return get


def list_keys(prefix: str, get: Getter) -> list[str]:
    """Every object key under a prefix of the public bucket (ListObjectsV2, paginated)."""
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        body = get(f"{S3}/?{httpx.QueryParams(params)}")
        if body is None:
            return keys
        root = ET.fromstring(body)
        keys += [e.text for e in root.findall("s3:Contents/s3:Key", _NS) if e.text]
        trunc = root.findtext("s3:IsTruncated", default="false", namespaces=_NS)
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=_NS)
        if trunc != "true" or not token:
            return keys


def assets_repo(metadata_yaml: str) -> str | None:
    """The `assets.repo` URL from metadata.yaml, if any (no YAML dependency: the block is flat)."""
    in_assets = False
    for line in metadata_yaml.splitlines():
        if re.match(r"^assets\s*:", line):
            in_assets = True
            continue
        if in_assets:
            if line and not line[0].isspace():
                break
            m = re.match(r"^\s+repo\s*:\s*['\"]?([^'\"\s#]+)", line)
            if m:
                return m.group(1)
    return None


def readme_resolved(readme: str) -> int | None:
    """The resolved count a submission's README states (e.g. 'Resolved 359 instances'), if it states one."""
    m = re.search(r"[Rr]esolved\s+(\d+)\s+instances", readme or "")
    return int(m.group(1)) if m else None


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def fetch_submission(name: str, out_root: str | Path, split: str = "verified", trajs: bool = False,
                     instances: set[str] | None = None, workers: int = 8, get: Getter | None = None,
                     git: Callable[[str, Path], None] | None = None, log=print) -> dict:
    """Download one submission into out_root/name (see module docstring). Returns a summary."""
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    client = None
    if get is None:
        client = httpx.Client(headers={"User-Agent": "keel-eval"})
        get = _default_getter(client)
    try:
        for rel in ("results/results.json", "metadata.yaml", "README.md"):
            data = get(f"{RAW}/{split}/{name}/{rel}")
            if data is None:
                if rel == "results/results.json":
                    raise FileNotFoundError(f"{split}/{name}: no results/results.json in swe-bench/experiments; "
                                            "check the submission name")
                continue
            _save(out / rel, data)
        resolved = json.loads((out / "results" / "results.json").read_text(encoding="utf-8")).get("resolved", [])
        meta = (out / "metadata.yaml").read_text(encoding="utf-8") if (out / "metadata.yaml").exists() else ""
        repo = assets_repo(meta)
        folders = ["logs", "trajs"] if trajs else ["logs"]
        if repo:
            n = _from_repo(repo, out, folders, instances, git or _git_clone, log)
            source = repo
        else:
            n = _from_s3(f"{split}/{name}", out, folders, instances, workers, get, log)
            source = f"s3://swe-bench-submissions/{split}/{name}"
    finally:
        if client is not None:
            client.close()
    readme = (out / "README.md").read_text(encoding="utf-8", errors="replace") if (out / "README.md").exists() else ""
    stated = readme_resolved(readme)
    n_patches = len(list(out.glob("logs/*/patch.diff")))
    summary = {"submission": name, "source": source, "files_downloaded": n, "patches": n_patches,
               "resolved": len(resolved), "readme_resolved": stated}
    if stated is not None and stated != len(resolved):
        log(f"warning: {name}: README says {stated} resolved, results.json lists {len(resolved)}")
    log(f"{name}: {n_patches} patches, {len(resolved)} resolved, {n} new files from {source}")
    return summary


def _wanted(key_rest: str, folder: str, instances: set[str] | None) -> bool:
    """key_rest is the part after '<split>/<name>/'."""
    parts = key_rest.split("/")
    if folder == "logs":
        if len(parts) != 3 or parts[2] not in LOG_FILES:
            return False
        iid = parts[1]
    else:
        iid = re.split(r"\.(?:traj|json|jsonl|md|txt)$", parts[-1])[0]
    return instances is None or iid in instances


def _from_s3(prefix: str, out: Path, folders: list[str], instances, workers: int, get: Getter, log) -> int:
    jobs = []
    for folder in folders:
        for key in list_keys(f"{prefix}/{folder}/", get):
            rest = key[len(prefix) + 1:]
            if _wanted(rest, folder, instances) and not (out / rest).exists():
                jobs.append((key, out / rest))
    if not jobs:
        return 0
    log(f"downloading {len(jobs)} files from s3://swe-bench-submissions/{prefix}")

    def one(job):
        key, dest = job
        data = get(f"{S3}/{key}")
        if data is not None:
            _save(dest, data)
            return 1
        return 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return sum(ex.map(one, jobs))


def _git_clone(url: str, dest: Path) -> None:
    subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(dest)], check=True)


def _from_repo(url: str, out: Path, folders: list[str], instances, git, log) -> int:
    log(f"cloning {url}")
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        git(url, Path(tmp) / "repo")
        src = Path(tmp) / "repo"
        for folder in folders:
            for p in (src / folder).rglob("*") if (src / folder).exists() else []:
                if not p.is_file():
                    continue
                rest = p.relative_to(src).as_posix()
                if _wanted(rest, folder, instances) and not (out / rest).exists():
                    (out / rest).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, out / rest)
                    n += 1
        preds = src / "all_preds.jsonl"
        if preds.exists() and not (out / "all_preds.jsonl").exists():
            shutil.copy2(preds, out / "all_preds.jsonl")
    return n


def save_instances(rows: list[dict], out: str | Path) -> int:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)
