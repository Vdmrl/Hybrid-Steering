"""Run the Qwen sweeps, then Falcon, once a GPU on this machine is free.

Scale 1 for rank 1, rank 2, and full rank injects the same total norm as the
full target-minus-source direction. Residual steering adds the raw last-token
residual at the layer chosen on a disjoint prompt split.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[2]
PY = Path(sys.executable)
RUNS = ROOT / "runs"
JUDGE_YAML = RUNS / "judge.yaml"
PLANS = RUNS / "plans"
LOGS = RUNS / "queue"
PAIRS = 0
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RESIDUAL_SCALES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEARCH_SCALES = [1.0, 4.0, 16.0]
CONCEPTS = (
    {
        "name": "theistic",
        "slug": "atheistic_framing-theistic_framing",
        "feature": "theistic_framing",
    },
    {
        "name": "fairytale",
        "slug": "fairytale",
        "feature": "fairytale",
        "source": "neutral",
        "target": "fairytale",
    },
    {"name": "russian", "slug": "en-ru", "feature": "russian_language"},
    {"name": "french", "slug": "en-fr", "feature": "french_language"},
    {"name": "chinese", "slug": "en-zh", "feature": "chinese_language"},
)
MODELS = (
    {"name": "qwen", "model": "Qwen/Qwen3.5-9B", "method": "gdn_add"},
    {"name": "falcon", "model": "tiiuae/Falcon-H1-7B-Instruct", "method": "mamba_add"},
)
EXTRAS = (
    "How do you tell if a houseplant needs water?",
    "What should someone check before lending a friend a book?",
    "Why do cities plant trees along streets?",
    "How can a person keep a bicycle chain from rusting?",
    "What happens to bread dough if the kitchen is cold?",
    "How would you choose a seat on a long train ride?",
    "Why do lakes freeze from the top?",
    "What is a practical way to remember where you parked?",
    "How can someone reduce noise from a dripping tap?",
    "Why do wool sweaters shrink in hot water?",
    "How would you pack a bag for a day hike?",
    "What makes ice cubes cloudy?",
    "How can a group split a restaurant bill fairly?",
    "Why do windows fog up in winter?",
    "What should you do with leftover rice so it stays safe to eat?",
    "How do crosswalk buttons change a traffic light?",
    "Why does metal feel colder than wood at the same temperature?",
    "How would you label boxes before moving apartments?",
    "What is a simple way to keep herbs fresh for a week?",
    "Why do some doors stick in humid weather?",
    "How can someone practice speaking clearly before a short talk?",
    "What should a person bring to a picnic if the forecast is uncertain?",
    "How do library due dates usually work?",
    "Why do kettles switch off when the water boils?",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(gpu: str | None = None) -> dict[str, str]:
    """Pass the server environment through. Do not invent cache or data paths."""
    env = os.environ.copy()
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "AI_HOME", "HF_HUB_CACHE"):
        if not env.get(name):
            raise RuntimeError(f"{name} is not set")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def data_dir() -> Path:
    return Path(os.environ["AI_HOME"]) / "benchmarks"


def log_call(argv: list[str], gpu: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = environment(gpu)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# gpu {gpu}: {' '.join(argv)}\n")
        stream.flush()
        return subprocess.call(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)


def free_gpus() -> list[str]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in text.splitlines():
        if not line.strip():
            continue
        index, used = (part.strip() for part in line.split(","))
        if int(used) < 4000:
            free.append(index)
    return free


def wait_gpus() -> list[str]:
    while True:
        free = free_gpus()
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} free GPUs: {', '.join(free) if free else 'none'}", flush=True)
        if free:
            return free
        time.sleep(60)


class Job:
    def __init__(self, name: str, function) -> None:
        self.name = name
        self.function = function

    def __call__(self, gpu: str) -> None:
        print(f"start {self.name} on GPU {gpu}", flush=True)
        self.function(gpu)
        print(f"done {self.name}", flush=True)


def run_pool(jobs: list[Job]) -> list[str]:
    gpus = wait_gpus()
    print(f"scheduling {len(jobs)} jobs on GPUs {gpus}", flush=True)
    slots: Queue[str] = Queue()
    for gpu in gpus:
        slots.put(gpu)
    failed: list[str] = []

    def worker(job: Job) -> None:
        gpu = slots.get()
        try:
            job(gpu)
        except Exception as error:
            failed.append(job.name)
            print(f"FAILED {job.name}: {error}", flush=True)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, jobs))
    return failed


def run_phase(jobs: list[Job]) -> None:
    failed = run_pool(jobs)
    if not failed:
        return
    print(f"retrying {failed}", flush=True)
    again = run_pool([job for job in jobs if job.name in set(failed)])
    if again:
        raise SystemExit(f"failed after retry: {again}")


def direction_ready(path: Path) -> bool:
    return (path / "direction.json").is_file() and (path / "direction.safetensors").is_file()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def prepare() -> dict:
    os.environ.update(environment())
    data = data_dir()
    RUNS.mkdir(parents=True, exist_ok=True)
    PLANS.mkdir(parents=True, exist_ok=True)
    if not JUDGE_YAML.exists():
        JUDGE_YAML.write_text(
            "model: openai/Qwen/Qwen3.8-27B\nmax_tokens: 2048\nthinking: false\n",
            encoding="utf-8",
        )
    import nltk

    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    sys.path.insert(0, str(data / "google-research"))
    from instruction_following_eval import evaluation_lib

    del evaluation_lib
    from huggingface_hub import snapshot_download

    for spec in MODELS:
        print(f"checking weights {spec['model']}", flush=True)
        snapshot_download(spec["model"])
    from hybrid_steering.detect import EVAL_QUESTIONS
    from hybrid_steering.direction import load_concept_pairs

    sys.path.insert(0, str(ROOT / "experiments/forgetting"))
    from questions import SIMPLE_QUESTIONS

    banned = set()
    for concept in CONCEPTS:
        print(f"reading pairs {concept['slug']}", flush=True)
        for row in load_concept_pairs(concept["slug"]):
            question = row.get("source_question") or row.get("user_prompt") or ""
            if question:
                banned.add(" ".join(str(question).lower().split()))
    kept = []
    seen = set()
    for question in (*EVAL_QUESTIONS, *SIMPLE_QUESTIONS, *EXTRAS):
        key = " ".join(question.lower().split())
        if key in banned or key in seen:
            continue
        seen.add(key)
        kept.append(question)
    if len(kept) < 16:
        raise SystemExit(f"only {len(kept)} judge prompts survive the extraction split")
    tune = [{"id": f"tune-{index:02d}", "prompt": text} for index, text in enumerate(kept[:12])]
    report = [{"id": f"eval-{index:02d}", "prompt": text} for index, text in enumerate(kept[12:])]
    tune_path, report_path = data / "judge-tune.jsonl", data / "judge-eval.jsonl"
    write_jsonl(tune_path, tune)
    write_jsonl(report_path, report)
    print(f"judge prompts: tune {len(tune)}, eval {len(report)}", flush=True)
    ifeval = data / "google-research/instruction_following_eval/data/input_data.jsonl"
    humaneval = data / "HumanEval.jsonl.gz"
    return {
        "tune": tune_path,
        "tune_sha": sha256(tune_path),
        "eval": report_path,
        "eval_sha": sha256(report_path),
        "benchmarks": {
            "ifeval": {
                "name": "ifeval",
                "path": str(ifeval),
                "sha256": sha256(ifeval),
                "evaluator": {
                    "root": str(data / "google-research"),
                    "evaluation_lib_sha256": sha256(
                        data / "google-research/instruction_following_eval/evaluation_lib.py"
                    ),
                },
            },
            "humaneval": {
                "name": "humaneval",
                "path": str(humaneval),
                "sha256": sha256(humaneval),
            },
        },
    }


def write_plan(path: Path, spec: dict, bench: dict, conditions: list[dict], prompts: dict) -> None:
    plan = {
        "model": spec["model"],
        "judge_dataset": {"path": str(prompts["eval"]), "sha256": prompts["eval_sha"]},
        "bench_dataset": bench,
        "batch_size": 32,
        "max_new_tokens": 1024,
        "conditions": conditions,
        "judge": {
            "feature": prompts["feature"],
            "config": str(JUDGE_YAML),
            "batch_size": 8,
        },
    }
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def run_config(path: Path, output: Path, gpu: str, log_path: Path) -> None:
    for batch in (32, 16, 8):
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["batch_size"] = batch
        path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        code = log_call(
            [
                str(PY),
                "experiments/pipeline/run.py",
                "run",
                "--config",
                str(path),
                "--output",
                str(output),
                "--run-judge",
            ],
            gpu,
            log_path,
        )
        if code == 0:
            return
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:].lower()
        if "out of memory" not in tail:
            raise RuntimeError(f"{path.name} failed, see {log_path}")
        print(f"{path.name} ran out of memory at batch {batch}", flush=True)
    raise RuntimeError(f"{path.name} ran out of memory at batch 8")


def steer_conditions(direction: Path, method: str) -> list[dict]:
    os.environ.update(environment())
    from hybrid_steering import load_direction
    from hybrid_steering.scoring import injected_norm

    tensor, *_ = load_direction(direction)
    full = injected_norm(tensor, None)
    conditions = [{"name": "baseline", "method": "baseline", "scales": [0]}]
    for name, rank in (("rank1", 1), ("rank2", 2), ("full", None)):
        gain = full / max(injected_norm(tensor, rank), 1e-8)
        row = {
            "name": name,
            "method": method,
            "direction": str(direction),
            "normalize": False,
            "prompt_position": -1,
            "gain": gain,
            "scales": SCALES,
        }
        if rank is not None:
            row["rank"] = rank
        conditions.append(row)
    return conditions


def extract_direction(concept: dict, spec: dict, gpu: str) -> Path:
    output = RUNS / "directions" / f"{spec['name']}-{concept['name']}"
    directory = output / "direction"
    if direction_ready(directory):
        return directory
    argv = [
        str(PY.parent / "hybrid-direction"),
        "--model",
        spec["model"],
        "--concept",
        concept["slug"],
        "--pairs",
        str(PAIRS),
        "--batch-size",
        "8",
        "--output",
        str(output),
    ]
    if concept.get("source"):
        argv += ["--source", concept["source"], "--target", concept["target"]]
    code = log_call(argv, gpu, LOGS / f"{spec['name']}-{concept['name']}-direction.log")
    if code or not direction_ready(directory):
        raise RuntimeError(f"direction failed for {spec['name']} {concept['name']}")
    return directory


def gdn_job(concept: dict, spec: dict, prompts: dict, gpu: str) -> None:
    prompts = {**prompts, "feature": concept["feature"]}
    direction = extract_direction(concept, spec, gpu)
    conditions = steer_conditions(direction, spec["method"])
    output = RUNS / "pipeline" / f"{spec['name']}-{concept['name']}"
    for bench_name, bench in prompts["benchmarks"].items():
        config = PLANS / f"{spec['name']}-{concept['name']}-{bench_name}.json"
        write_plan(config, spec, bench, conditions, prompts)
        run_config(config, output, gpu, LOGS / f"{spec['name']}-{concept['name']}-{bench_name}.log")


def residual_job(concept: dict, spec: dict, prompts: dict, gpu: str) -> None:
    prompts = {**prompts, "feature": concept["feature"]}
    output = RUNS / "directions" / f"{spec['name']}-{concept['name']}-residual"
    directory = output / "direction"
    log = LOGS / f"{spec['name']}-{concept['name']}-residual.log"
    if not direction_ready(directory):
        argv = [
            str(PY),
            "experiments/pipeline/residual.py",
            "extract",
            "--model",
            spec["model"],
            "--concept",
            concept["slug"],
            "--pairs",
            str(PAIRS),
            "--batch-size",
            "8",
            "--output",
            str(output),
        ]
        if concept.get("source"):
            argv += ["--source", concept["source"], "--target", concept["target"]]
        if log_call(argv, gpu, log) or not direction_ready(directory):
            raise RuntimeError(f"residual extract failed for {concept['name']}")
    choice = output / "layer_choice.json"
    if not choice.exists():
        code = log_call(
            [
                str(PY),
                "experiments/pipeline/residual.py",
                "pick-layer",
                "--model",
                spec["model"],
                "--direction",
                str(directory),
                "--feature",
                concept["feature"],
                "--prompts",
                str(prompts["tune"]),
                "--scales",
                *[str(scale) for scale in SEARCH_SCALES],
                "--judge-config",
                str(JUDGE_YAML),
                "--output",
                str(output),
            ],
            gpu,
            log,
        )
        if code or not choice.exists():
            raise RuntimeError(f"layer search failed for {concept['name']}")
    layer = int(json.loads(choice.read_text(encoding="utf-8"))["chosen"]["layer"])
    conditions = [
        {"name": "baseline", "method": "baseline", "scales": [0]},
        {
            "name": f"residual-L{layer}",
            "method": "residual_add",
            "direction": str(directory),
            "layer": layer,
            "normalize": False,
            "scales": RESIDUAL_SCALES,
        },
    ]
    result = RUNS / "pipeline" / f"{spec['name']}-{concept['name']}-residual"
    for bench_name, bench in prompts["benchmarks"].items():
        config = PLANS / f"{spec['name']}-{concept['name']}-residual-{bench_name}.json"
        write_plan(config, spec, bench, conditions, prompts)
        run_config(config, result, gpu, LOGS / f"{config.stem}.log")


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    pid = LOGS / "queue.pid"
    if pid.exists() and Path(f"/proc/{pid.read_text().strip()}").exists():
        raise SystemExit(f"queue already running as {pid.read_text().strip()}")
    pid.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    prompts = prepare()
    print("data is ready, waiting for a free GPU", flush=True)
    for spec in MODELS:
        run_phase(
            [
                Job(
                    f"{spec['name']}-{concept['name']}",
                    lambda gpu, c=concept, s=spec: gdn_job(c, s, prompts, gpu),
                )
                for concept in CONCEPTS
            ]
        )
        run_phase(
            [
                Job(
                    f"{spec['name']}-{concept['name']}-residual",
                    lambda gpu, c=concept, s=spec: residual_job(c, s, prompts, gpu),
                )
                for concept in CONCEPTS
            ]
        )
    print(f"finished. reports are under {RUNS / 'pipeline'}", flush=True)


if __name__ == "__main__":
    main()
