import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def bash():
    if os.name == "nt":
        git = shutil.which("git")
        path = Path(git).parents[1] / "bin/bash.exe" if git else None
    else:
        executable = shutil.which("bash")
        path = Path(executable) if executable else None
    if not path or not path.is_file():
        pytest.skip("Bash is required for shell contract tests")
    return str(path)


@pytest.mark.parametrize("gpus", ["0", "0,1"])
def test_drop_order_and_gpu_assignment(tmp_path, bash, gpus):
    stub = tmp_path / "launcher"
    stub.write_text(
        """#!/usr/bin/env bash
set -eu
while (($#)); do
  case $1 in
    --output) job=${2##*/}; shift ;;
    --gpus) gpus=$2; shift ;;
  esac
  shift
done
printf '%s %s start\n' "$job" "$gpus" >> "$TEST_LOG"
case $job in
  job1_mega)
    if [[ $TEST_GPUS == *,* ]]; then
      for i in {1..80}; do
        grep -q '^job2 1 start$' "$TEST_LOG" && break
        sleep 0.05
      done
      grep -q '^job2 1 start$' "$TEST_LOG"
    fi
    ;;
  job2)
    if [[ $TEST_GPUS != *,* ]]; then grep -q '^job1_mega 0 merged$' "$TEST_LOG"; fi
    ;;
  job4a)
    grep -q '^job1_mega 0 merged$' "$TEST_LOG"
    grep -q '^job2 .* merged$' "$TEST_LOG"
    ;;
esac
printf '%s %s merged\n' "$job" "$gpus" >> "$TEST_LOG"
""",
        encoding="utf-8",
        newline="\n",
    )
    stub.chmod(0o755)
    log = tmp_path / "launched.log"
    result = subprocess.run(
        [bash, str(PACKAGE / "jobs/run_drop1_h200.sh")],
        env={
            **os.environ,
            "MODEL": "/unused",
            "GPUS": gpus,
            "PYTHON_BIN": stub.as_posix(),
            "TEST_GPUS": gpus,
            "TEST_LOG": log.as_posix(),
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    entries = log.read_text().splitlines()
    assert ("job2 1 start" if "," in gpus else "job2 0 start") in entries
    assert entries[-2:] == [f"job4a {gpus} start", f"job4a {gpus} merged"]
    assert "Return outputs/job1_mega now" in result.stdout


@pytest.mark.parametrize("gpus", ["2", "0,1", "2,4,7"])
def test_drop2_jobs_share_all_gpus_in_order(tmp_path, bash, gpus):
    stub = tmp_path / "launcher"
    stub.write_text(
        """#!/usr/bin/env bash
set -eu
resume=no
while (($#)); do
  case $1 in
    --output) job=${2##*/}; shift ;;
    --gpus) gpus=$2; shift ;;
    --resume) resume=yes ;;
  esac
  shift
done
[[ $resume == yes ]]
printf '%s %s\n' "$job" "$gpus" >> "$TEST_LOG"
""",
        encoding="utf-8",
        newline="\n",
    )
    stub.chmod(0o755)
    log = tmp_path / "drop2.log"
    subprocess.run(
        [bash, str(PACKAGE / "jobs/run_drop2.sh"), "--resume"],
        env={
            **os.environ,
            "MODEL": "/unused",
            "GPUS": gpus,
            "PYTHON_BIN": stub.as_posix(),
            "TEST_LOG": log.as_posix(),
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert log.read_text().splitlines() == [
        f"{job} {gpus}"
        for job in (
            "job2b",
            "job1b",
            "job4b",
            "job3_best_multilayer",
            "job3b_gdn24_ctrl",
            "job1c_gdn24_micro",
        )
    ]


@pytest.mark.parametrize("mode", ["local", "github"])
def test_fetch_validates_before_install(tmp_path, bash, mode):
    directions = tmp_path / "package/directions"
    directions.mkdir(parents=True)
    script = directions / "fetch.sh"
    shutil.copyfile(PACKAGE / "directions/fetch.sh", script)
    source = tmp_path / "external direction files"
    checksums = []
    for method in ("caa24", "hys"):
        (source / method).mkdir(parents=True)
        for tag in "NFTRP":
            file = Path(method) / f"{tag}.safetensors"
            payload = str(file).encode()
            (source / file).write_bytes(payload)
            checksums.append(
                f"{hashlib.sha256(payload).hexdigest()}  directions/{file.as_posix()}\n"
            )
    (directions / "SHA256SUMS.txt").write_text("".join(checksums), encoding="ascii", newline="\n")
    cmd = [bash, str(script)]
    env = os.environ.copy()
    if mode == "local":
        cmd.extend(("--from-dir", source.name))
    else:
        gh = tmp_path / "gh"
        gh.write_text(
            """#!/usr/bin/env bash
set -eu
[[ $1 == release && $2 == download && $GITHUB_TOKEN == test-only ]]
while (($#)); do
  if [[ $1 == --dir ]]; then dest=$2; break; fi
  shift
done
for method in caa24 hys; do
  for tag in N F T R P; do
    cp "$TEST_SOURCE/$method/$tag.safetensors" "$dest/$method-$tag.safetensors"
  done
done
""",
            encoding="utf-8",
            newline="\n",
        )
        gh.chmod(0o755)
        cmd.extend(("--github-release", "test-org/test-repo", "test-tag"))
        env.update(
            PATH=f"{tmp_path}{os.pathsep}{env['PATH']}",
            TEST_SOURCE=source.as_posix(),
            GITHUB_TOKEN="test-only",
        )
    subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, check=True, timeout=30)
    installed = directions / "caa24/N.safetensors"
    good = installed.read_bytes()
    (source / "caa24/N.safetensors").write_bytes(b"not a valid tensor")
    failed = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert failed.returncode != 0
    assert installed.read_bytes() == good
