"""Run an entry the way the organizer reruns it (docker) or locally (quick, non-authoritative).

Docker mode: the image named in reproduce.json (`python`) maps to PYTHON_BASE_TAG.
A pinned requirements.txt is installed at image BUILD time into a derived image
cached by the hash of its contents. The run itself uses --network none, read-only
code/data mounts, a 128 MiB /tmp tmpfs and a 32 MiB /output tmpfs. The container is
started detached (`sleep infinity`), the command runs via `docker exec` under
`timeout`, the CSV is read out with `docker exec cat` while the container is up
(a stopped container loses its tmpfs), and RunResult.limits records what
`docker inspect` reports as applied.

Local mode: the same argv with the container path prefixes (/data, /submission,
/output, /tmp, /main_data) rewritten to host paths. Nothing is copied and
requirements are not installed, so it cannot prove an offline rerun works.

Timeouts: exit_code is TIMEOUT_EXIT_CODE (124, as timeout(1) reports) and the
process or container is killed. wall_s covers only the run, not the image build.
Callers own cleanup of the temporary output directory that csv_path lives in.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PYTHON_BASE_TAG = "python:3.13-slim"
DERIVED_REPOSITORY = "eval-suite/deps"
TIMEOUT_EXIT_CODE = 124
TAIL_CHARS = 4000
BUILD_TIMEOUT_S = 1800
PULL_TIMEOUT_S = 1800
DOCKER_FALLBACK = "/usr/local/bin/docker"
TMP_TMPFS_BYTES = 128 * 2**20      # ROB-TMP-QUOTA: /tmp is a 128 MiB tmpfs
OUTPUT_TMPFS_BYTES = 32 * 2**20   # ROB-OUTPUT-QUOTA: /output is a 32 MiB tmpfs
_MEM_UNITS = {"B": 1, "kB": 1e3, "KB": 1e3, "KiB": 2**10, "MB": 1e6, "MiB": 2**20,
              "GB": 1e9, "GiB": 2**30, "TB": 1e12, "TiB": 2**40}


class EntryError(RuntimeError):
    """The entry cannot be run at all (bad reproduce.json, no docker daemon, failed build)."""


@dataclass
class RunResult:
    csv_path: Path | None
    wall_s: float
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    peak_mem_mb: float | None
    mode: str
    limits: dict = field(default_factory=dict)  # docker: what docker inspect reports as applied; local: {}


def _tail(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    return data[-TAIL_CHARS:]


def _docker_bin() -> str:
    found = shutil.which("docker") or (DOCKER_FALLBACK if Path(DOCKER_FALLBACK).exists() else None)
    if not found:
        raise EntryError("docker CLI not found on PATH or at /usr/local/bin/docker")
    return found


def docker_available() -> tuple[bool, str]:
    """Return (ok, message). `docker info` fails when the daemon is not running."""
    try:
        docker = _docker_bin()
    except EntryError as exc:
        return False, str(exc)
    try:
        proc = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
                              capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker info failed: {exc}"
    if proc.returncode != 0:
        return False, "docker daemon not reachable: " + _tail(proc.stderr).strip()
    return True, f"docker server {proc.stdout.strip()}"


def _existing_dir(path, label) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise EntryError(f"{label} is not a directory: {resolved}")
    return resolved


def _load_config(entry: Path) -> dict:
    path = entry / "reproduce.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EntryError(f"cannot read {path}: {exc}") from exc
    command = config.get("command") if isinstance(config, dict) else None
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise EntryError("reproduce.json needs a nonempty command list of strings")
    if config.get("image") != "python":
        raise EntryError(f"unsupported image {config.get('image')!r}; only 'python' is mapped")
    return config


def _requirements_bytes(entry: Path) -> bytes | None:
    """Raw requirements.txt bytes, or None when it has no installable lines."""
    path = entry / "requirements.txt"
    if not path.is_file():
        return None
    data = path.read_bytes()
    lines = [line.strip() for line in data.decode("utf-8", "replace").splitlines()]
    return data if any(line and not line.startswith("#") for line in lines) else None


def _image_exists(docker: str, tag: str) -> bool:
    proc = subprocess.run([docker, "image", "inspect", tag], capture_output=True,
                          stdin=subprocess.DEVNULL)
    return proc.returncode == 0


def _ensure_base(docker: str, tag: str) -> None:
    if _image_exists(docker, tag):
        return
    try:
        proc = subprocess.run([docker, "pull", tag], capture_output=True, text=True,
                              timeout=PULL_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise EntryError(f"pulling {tag} timed out") from exc
    if proc.returncode != 0:
        raise EntryError(f"pulling {tag} failed: {_tail(proc.stderr).strip()}")


def _ensure_image(entry: Path, docker: str) -> str:
    """Return the image tag to run: the base image, or a derived image with requirements baked in."""
    requirements = _requirements_bytes(entry)
    if requirements is None:
        _ensure_base(docker, PYTHON_BASE_TAG)
        return PYTHON_BASE_TAG
    digest = hashlib.sha256(PYTHON_BASE_TAG.encode() + b"\n" + requirements).hexdigest()[:16]
    tag = f"{DERIVED_REPOSITORY}:{digest}"
    if _image_exists(docker, tag):
        return tag
    context = Path(tempfile.mkdtemp(prefix="eval-build-"))
    try:
        (context / "requirements.txt").write_bytes(requirements)
        (context / "Dockerfile").write_text(
            f"FROM {PYTHON_BASE_TAG}\n"
            "COPY requirements.txt /tmp/eval-requirements.txt\n"
            "RUN pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore "
            "-r /tmp/eval-requirements.txt && rm /tmp/eval-requirements.txt\n",
            encoding="utf-8")
        try:
            proc = subprocess.run([docker, "build", "-q", "-t", tag, str(context)],
                                  capture_output=True, text=True, timeout=BUILD_TIMEOUT_S,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise EntryError("building the dependency image timed out") from exc
    finally:
        shutil.rmtree(context, ignore_errors=True)
    if proc.returncode != 0:
        raise EntryError("building the dependency image failed:\n" + _tail(proc.stderr or proc.stdout))
    return tag


def _sample_mem_mb(docker: str, name: str) -> float | None:
    try:
        proc = subprocess.run([docker, "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                              capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    match = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", proc.stdout)
    if not match or match.group(2) not in _MEM_UNITS:
        return None
    return float(match.group(1)) * _MEM_UNITS[match.group(2)] / 2**20


class _MemorySampler:
    """Polls `docker stats` while the container runs. Peak is None if no sample succeeded."""

    def __init__(self, docker: str, name: str):
        self._docker, self._name = docker, name
        self._stop = threading.Event()
        self._peak: float | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            mb = _sample_mem_mb(self._docker, self._name)
            if mb is not None and (self._peak is None or mb > self._peak):
                self._peak = mb
            self._stop.wait(0.5)

    def stop(self) -> float | None:
        self._stop.set()
        self._thread.join(timeout=15)
        return self._peak


def _docker_quiet(docker: str, args: list, timeout: int = 60):
    try:
        return subprocess.run([docker, *args], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _applied_limits(docker: str, name: str, requested: dict) -> dict:
    """What docker reports for the container: the limits and mounts that were actually applied."""
    proc = _docker_quiet(docker, ["inspect", name])
    try:
        info = json.loads(proc.stdout)[0]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return {**requested, "source": "requested (docker inspect failed)"}
    host = info.get("HostConfig") or {}
    nano = host.get("NanoCpus") or 0
    return {
        "source": "docker inspect",
        "cpus": nano / 1e9 if nano else None,
        "memory_bytes": host.get("Memory"),
        "memory_swap_bytes": host.get("MemorySwap"),
        "network": host.get("NetworkMode"),
        "working_dir": (info.get("Config") or {}).get("WorkingDir"),
        "tmpfs": host.get("Tmpfs") or {},
        "tmpfs_size_bytes": {m.get("Target"): (m.get("TmpfsOptions") or {}).get("SizeBytes")
                             for m in host.get("Mounts") or [] if m.get("Type") == "tmpfs"},
        "mounts": [{"destination": m.get("Destination"), "type": m.get("Type"), "rw": m.get("RW")}
                   for m in info.get("Mounts") or []],
    }


def _copy_output(docker: str, name: str, out: Path) -> None:
    """Stream /output/submission.csv out of the running container, if the run wrote one.

    docker cp cannot see a --mount tmpfs and a stopped container loses its tmpfs, so the
    file is read with exec while the container is still up.
    """
    try:
        proc = subprocess.run([docker, "exec", name, "cat", "/output/submission.csv"],
                              capture_output=True, timeout=300, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return
    if proc.returncode == 0:
        (out / "submission.csv").write_bytes(proc.stdout)


def _run_docker(entry, data, main_data, out, command, image, *, docker, timeout_s, cpus, memory_gb):
    """Run the command in a detached container with the organizer's limits, then remove it.

    The container starts with `sleep infinity`; the reproduce command runs through `docker exec`
    under `timeout`, so the CSV can be read out and the applied limits inspected while the
    container is still up. /tmp (128 MiB) and /output (32 MiB) are tmpfs mounts that start empty.
    """
    name = f"eval-{uuid.uuid4().hex[:12]}"
    memory = f"{memory_gb:g}g"
    requested = {"cpus": cpus, "memory_bytes": int(memory_gb * 2**30), "memory_swap_bytes": int(memory_gb * 2**30),
                 "network": "none", "tmpfs": {"/tmp": TMP_TMPFS_BYTES}, "output_tmpfs_bytes": OUTPUT_TMPFS_BYTES,
                 "working_dir": "/submission"}
    argv = [docker, "run", "-d", "--name", name, "--network", "none",
            "--cpus", str(cpus), "--memory", memory, "--memory-swap", memory,
            "--tmpfs", f"/tmp:rw,size={TMP_TMPFS_BYTES}",
            "--mount", f"type=tmpfs,destination=/output,tmpfs-size={OUTPUT_TMPFS_BYTES}",
            "-v", f"{entry}:/submission:ro", "-v", f"{data}:/data:ro"]
    if main_data is not None:
        argv += ["-v", f"{main_data}:/main_data:ro"]
    argv += ["-w", "/submission", image, "sleep", "infinity"]
    try:
        started_proc = subprocess.run(argv, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EntryError(f"could not start the container: {exc}") from exc
    if started_proc.returncode != 0:
        raise EntryError("could not start the container: " + _tail(started_proc.stderr).strip())
    sampler = _MemorySampler(docker, name)
    try:
        sampler.start()
        exec_argv = [docker, "exec", "-w", "/submission", name,
                     "timeout", "--kill-after=10", str(timeout_s), *command]
        started = time.monotonic()
        try:
            proc = subprocess.run(exec_argv, capture_output=True, timeout=timeout_s + 60, stdin=subprocess.DEVNULL)
            exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            _docker_quiet(docker, ["kill", name])
            exit_code = TIMEOUT_EXIT_CODE
            stdout = exc.stdout or b""
            stderr = (exc.stderr or b"") + f"\n[killed after {timeout_s}s timeout]".encode()
        wall = time.monotonic() - started
        peak = sampler.stop()
        limits = _applied_limits(docker, name, requested)
        _copy_output(docker, name, out)
    finally:
        sampler.stop()
        _docker_quiet(docker, ["rm", "-f", name])
    return exit_code, stdout, stderr, wall, peak, limits


def _rewrite(arg: str, mapping) -> str:
    """Replace a container path prefix with its host path, as a whole argument or after `--opt=`."""
    for prefix, target in mapping:
        if arg == prefix or arg.startswith(prefix + "/"):
            return str(target) + arg[len(prefix):]
    key, sep, value = arg.partition("=")
    if sep:
        for prefix, target in mapping:
            if value == prefix or value.startswith(prefix + "/"):
                return f"{key}={target}{value[len(prefix):]}"
    return arg


def _run_local(entry, data, main_data, out, tmp, command, *, timeout_s):
    mapping = [("/data", data), ("/submission", entry), ("/output", out), ("/tmp", tmp)]
    if main_data is not None:
        mapping.append(("/main_data", main_data))
    argv = [_rewrite(arg, mapping) for arg in command]
    if argv[0] == "python":
        argv[0] = sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=entry, capture_output=True, timeout=timeout_s, env=env,
                              stdin=subprocess.DEVNULL)
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = TIMEOUT_EXIT_CODE
        stdout = exc.stdout or b""
        stderr = (exc.stderr or b"") + f"\n[killed after {timeout_s}s timeout]".encode()
    wall = time.monotonic() - started
    return exit_code, stdout, stderr, wall, None


_IMAGE_LOCK = threading.Lock()


def run_entry(entry_dir, data_root, *, main_data_root=None, mode="docker", timeout_s=1800,
              cpus=2, memory_gb=16) -> RunResult:
    """Run an entry's reproduce.json command and report what it produced.

    data_root is a dataset root containing test/ (mounted at /data). main_data_root,
    when given, is mounted at /main_data (adaptation round). Raises EntryError when
    the entry cannot be run at all; a failing command is reported in the result.
    """
    if mode not in ("docker", "local"):
        raise EntryError(f"mode must be 'docker' or 'local', not {mode!r}")
    if timeout_s <= 0:
        raise EntryError("timeout_s must be positive")
    entry = _existing_dir(entry_dir, "entry_dir")
    data = _existing_dir(data_root, "data_root")
    main = _existing_dir(main_data_root, "main_data_root") if main_data_root is not None else None
    config = _load_config(entry)
    command = config["command"]

    if mode == "docker":
        ok, message = docker_available()
        if not ok:
            raise EntryError(f"docker mode needs a running Docker daemon ({message}). "
                             "Use mode='local' for a non-authoritative run.")
        docker = _docker_bin()
        with _IMAGE_LOCK:  # concurrent --jobs runs must not build or pull the same tag twice at once
            image = _ensure_image(entry, docker)

    out = Path(tempfile.mkdtemp(prefix="eval-out-"))
    limits: dict = {}
    if mode == "docker":
        exit_code, stdout, stderr, wall, peak, limits = _run_docker(
            entry, data, main, out, command, image,
            docker=docker, timeout_s=timeout_s, cpus=cpus, memory_gb=memory_gb)
    else:
        tmp = Path(tempfile.mkdtemp(prefix="eval-tmp-"))
        exit_code, stdout, stderr, wall, peak = _run_local(
            entry, data, main, out, tmp, command, timeout_s=timeout_s)
    csv = out / "submission.csv"
    return RunResult(
        csv_path=csv if csv.is_file() else None,
        wall_s=wall,
        exit_code=exit_code,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
        peak_mem_mb=peak,
        mode=mode,
        limits=limits,
    )
