"""Oracle 2 in Docker: build the keel-oracle2 image and run the SWE-ABS harness in it.

The harness talks to Docker through the Python SDK and starts one container per SWE-bench
instance. Run inside our image, it reaches the HOST daemon through the mounted socket, so those
instance containers are siblings on the host ("Docker-out-of-Docker"); patches and tests are copied
into them through the Docker API, so no host paths need to be shared with them. Only three things are
mounted into the harness container:

  <models_root>  -> /data/submissions                       the `keel eval fetch` folder (read-only)
  <results_dir>  -> /opt/SWE-ABS/swe-bench/swe_abs_res      final_results.json lands here
  <results_dir>/logs -> /opt/SWE-ABS/swe-bench/logs         per-instance harness logs
  volume keel-hf-cache -> /cache/hf                          Hugging Face cache (the ABS dataset)

Output: <results_dir>/eval_agent/<run_id>/final_results.json, read by `keel eval abs-labels
--results-dir <results_dir> --run-id <run_id>`.

Works on a Linux host, and on Windows / macOS (Intel) with Docker Desktop's Linux engine; the
SWE-bench instance images are x86-64, so ARM hosts are refused unless forced.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .abs_labels import ABS_DATASET

IMAGE = "keel-oracle2"
HF_VOLUME = "keel-hf-cache"
CONTAINER_MODELS = "/data/submissions"
CONTAINER_RESULTS = "/opt/SWE-ABS/swe-bench/swe_abs_res"
CONTAINER_LOGS = "/opt/SWE-ABS/swe-bench/logs"
MIN_FREE_GB = 120  # SWE-bench's own guidance for the harness
DOCKERFILE = Path("docker") / "oracle2" / "Dockerfile"


@dataclass
class Preflight:
    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def preflight(image: str = IMAGE, results_dir: str | Path | None = None, allow_arm: bool = False,
              runner=_run) -> Preflight:
    """Check the Docker CLI, the engine (Linux, x86-64, resources), disk space and the image."""
    pf = Preflight(ok=True)
    if shutil.which("docker") is None and runner is _run:
        pf.ok = False
        pf.problems.append("the docker command is not installed (install Docker Engine or Docker Desktop)")
        return pf
    try:
        r = runner(["docker", "info", "--format", "{{json .}}"])
    except (OSError, subprocess.TimeoutExpired) as e:
        pf.ok = False
        pf.problems.append(f"could not run `docker info`: {e}")
        return pf
    if r.returncode != 0:
        pf.ok = False
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        pf.problems.append("the Docker engine is not running or not reachable (start Docker Desktop or the docker "
                           "service)" + (f": {detail[-1]}" if detail else ""))
        return pf
    try:
        info = json.loads(r.stdout)
    except json.JSONDecodeError:
        info = {}
    arch = str(info.get("Architecture", "")).lower()
    ostype = str(info.get("OSType", "")).lower()
    ncpu = int(info.get("NCPU") or 0)
    mem_gb = round(int(info.get("MemTotal") or 0) / 1e9, 1)
    pf.info = {"os_type": ostype, "architecture": arch, "cpus": ncpu, "memory_gb": mem_gb,
               "docker_root": info.get("DockerRootDir"), "server_version": info.get("ServerVersion")}
    if ostype and ostype != "linux":
        pf.ok = False
        pf.problems.append(f"the engine runs {ostype} containers; switch Docker Desktop to Linux containers")
    if arch and arch not in ("x86_64", "amd64"):
        msg = (f"the engine is {arch}; SWE-bench instance images are x86-64 and give wrong or failing results "
               "under emulation")
        if allow_arm:
            pf.warnings.append(msg)
        else:
            pf.ok = False
            pf.problems.append(msg + " (use an x86-64 machine, or pass --allow-arm to try anyway)")
    if ncpu and ncpu < 4:
        pf.warnings.append(f"only {ncpu} CPUs for Docker; 8 or more is recommended")
    if mem_gb and mem_gb < 15:
        pf.warnings.append(f"only {mem_gb} GB memory for Docker; 16 GB or more is recommended "
                           "(Docker Desktop: Settings -> Resources)")
    if results_dir is not None:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(p).free / 1e9
        pf.info["free_gb_results_disk"] = round(free, 1)
        if free < MIN_FREE_GB:
            pf.warnings.append(f"{free:.0f} GB free where results go; the instance images need about "
                               f"{MIN_FREE_GB} GB on the Docker disk (check Docker Desktop's disk image size too)")
    img = runner(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    pf.info["image_present"] = img.returncode == 0
    if img.returncode != 0:
        pf.ok = False
        pf.problems.append(f"image {image!r} not found; build it with `keel eval abs-docker-build`")
    return pf


def repo_root() -> Path | None:
    """The keel source checkout (needed for `docker build`), if Keel is installed from one."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / DOCKERFILE).exists() and (root / "pyproject.toml").exists() else None


def build_command(tag: str = IMAGE, sweabs_ref: str | None = None, lfs: bool = False,
                  root: Path | None = None, no_cache: bool = False) -> list[str]:
    root = root or repo_root()
    if root is None:
        raise FileNotFoundError("docker/oracle2/Dockerfile not found: install Keel from a source checkout "
                                "(pip install -e <keel repo>) to build the image")
    cmd = ["docker", "build", "-f", str(root / DOCKERFILE), "-t", tag]
    if sweabs_ref:
        cmd += ["--build-arg", f"SWEABS_REF={sweabs_ref}"]
    if lfs:
        cmd += ["--build-arg", "SWEABS_LFS=1"]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(str(root))
    return cmd


def _bind(src: Path, dst: str, readonly: bool = False) -> list[str]:
    spec = f"type=bind,source={src.resolve()},target={dst}"
    return ["--mount", spec + (",readonly" if readonly else "")]


def run_command(models_root: str | Path, models: list[str], run_id: str, results_dir: str | Path,
                instance_ids: list[str] | None = None, workers: int = 8, timeout: int = 1800,
                image: str = IMAGE, rerun: bool = False, socket: str = "/var/run/docker.sock",
                host_uid_gid: tuple[int, int] | None = None, name: str | None = None) -> list[str]:
    """`docker run ...` for the harness. Pure: builds the argument list, runs nothing."""
    models_root, results_dir = Path(models_root), Path(results_dir)
    harness = ["python", "-m", "swebench.runtest.run_evaluation_test",
               "--predictions_test_path", ABS_DATASET,
               "--vaild_model_path", CONTAINER_MODELS,
               "--vaild_model_name", ",".join(models),
               "--run_id", run_id, "--max_workers", str(workers), "--timeout", str(timeout)]
    if instance_ids:
        harness += ["--instance_ids", ",".join(instance_ids)]
    if rerun:
        harness += ["--re_run_eval", "True"]
    cmd = ["docker", "run", "--rm", "--init",
           "-v", f"{socket}:/var/run/docker.sock",
           *_bind(models_root, CONTAINER_MODELS, readonly=True),
           *_bind(results_dir, CONTAINER_RESULTS),
           *_bind(results_dir / "logs", CONTAINER_LOGS),
           "--mount", f"type=volume,source={HF_VOLUME},target=/cache/hf",
           "--ulimit", "nofile=4096:4096"]  # the harness raises RLIMIT_NOFILE to 4096 itself
    if name:
        cmd += ["--name", name]
    cmd.append(image)
    if host_uid_gid:  # give the results back to the host user (files are written as root inside)
        uid, gid = host_uid_gid
        quoted = " ".join(_sh_quote(a) for a in harness)
        cmd += ["sh", "-c", f"{quoted}; rc=$?; chown -R {uid}:{gid} {CONTAINER_RESULTS} {CONTAINER_LOGS}; exit $rc"]
    else:
        cmd += harness
    return cmd


def _sh_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def host_ids() -> tuple[int, int] | None:
    """(uid, gid) on POSIX hosts, so the harness can chown its output back; None on Windows."""
    if hasattr(os, "getuid") and os.name == "posix":
        return os.getuid(), os.getgid()
    return None


def final_results_path(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / "eval_agent" / run_id / "final_results.json"


def execute(cmd: list[str], log=print) -> int:
    log("running: " + " ".join(cmd))
    return subprocess.run(cmd).returncode
