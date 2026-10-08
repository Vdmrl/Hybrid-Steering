"""Run the Qwen sweeps, then Falcon, once a GPU on this machine is free.

Generation holds the card. Scoring starts only after that process exits, with
no CUDA device, so the next concept can generate while the previous one is judged.

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
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[2]
PY = Path(sys.executable)
RUNS = ROOT / "runs"
JUDGE_YAML = RUNS / "judge.yaml"
PLANS = RUNS / "plans"
LOGS = RUNS / "queue"
PAIRS = 0
GENERATE_BATCHES = (256, 128, 64)
EXTRACT_BATCHES = (64,)
# Same ladder as Qwen. mamba-ssm supplies the fused scan, so the eager 192 GiB
# tensor is gone. The last step is the batch that still fits without that kernel.
FALCON_GENERATE_BATCHES = (256, 128, 64, 16)
FALCON_EXTRACT_BATCHES = (64, 16)
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RESIDUAL_SCALES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LANGUAGE_LAYERS = (
    {"name": "russian", "slug": "en-ru", "feature": "russian_language", "layer": 8},
    {"name": "french", "slug": "en-fr", "feature": "french_language", "layer": 20},
    {"name": "chinese", "slug": "en-zh", "feature": "chinese_language", "layer": 12},
    {"name": "arabic", "slug": "en-ar", "feature": "arabic_language", "layer": 16},
)
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
    for name in ("AI_HOME", "HF_HUB_CACHE"):
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


def log_call(argv: list[str], gpu: str | None, log_path: Path) -> int:
    """Run one command. ``gpu=None`` hides every CUDA device so scoring cannot hold a card."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = environment(gpu)
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    label = "cpu" if gpu is None else f"gpu {gpu}"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {label}: {' '.join(argv)}\n")
        stream.flush()
        return subprocess.call(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)


def allowed_gpus() -> set[str]:
    raw = os.environ.get("GPU_ALLOWLIST", "")
    allowed = {part.strip() for part in raw.split(",") if part.strip()}
    if not allowed:
        raise RuntimeError("GPU_ALLOWLIST is not set")
    return allowed


def free_gpus() -> list[str]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    allowed = allowed_gpus()
    free = []
    for line in text.splitlines():
        if not line.strip():
            continue
        index, used = (part.strip() for part in line.split(","))
        # A full card is about 144 GiB. 19 GiB left by another job still fits this model.
        limit = int(os.environ.get("GPU_FREE_MIB", "90000"))
        if index in allowed and int(used) < limit:
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
    known: set[str] = set(gpus)
    known_lock = threading.Lock()
    for gpu in gpus:
        slots.put(gpu)
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(30):
            for gpu in free_gpus():
                with known_lock:
                    if gpu in known:
                        continue
                    known.add(gpu)
                slots.put(gpu)
                print(f"GPU {gpu} joined the pool", flush=True)

    threading.Thread(target=watch, daemon=True).start()
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

    try:
        with ThreadPoolExecutor(max_workers=max(len(gpus), len(allowed_gpus()))) as pool:
            list(pool.map(worker, jobs))
    finally:
        stop.set()
    return failed


def run_phase(jobs: list[Job]) -> None:
    failed = run_pool(jobs)
    again: list[str] = []
    if failed:
        print(f"retrying {failed}", flush=True)
        again = run_pool([job for job in jobs if job.name in set(failed)])
    # Cards are already back in the pool. Scoring uses the CPU and the remote judge.
    score_failed = wait_scores()
    if score_failed:
        print(f"retrying scores {[path.name for path, _, _ in score_failed]}", flush=True)
        for path, output, log_path in score_failed:
            submit_score(path, output, log_path)
        score_failed = wait_scores()
    if again:
        print(f"still failed after retry: {again}", flush=True)
    if score_failed:
        print(f"scores still failed: {[path.name for path, _, _ in score_failed]}", flush=True)


def direction_ready(path: Path) -> bool:
    return (path / "direction.json").is_file() and (path / "direction.safetensors").is_file()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def prepare(models: tuple[dict, ...] | None = None) -> dict:
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

    for spec in MODELS if models is None else models:
        print(f"checking weights {spec['model']}", flush=True)
        snapshot_download(spec["model"])
    tune_path, report_path = data / "judge-tune.jsonl", data / "judge-eval.jsonl"
    if tune_path.exists() and report_path.exists():
        print(f"reusing prompts {tune_path.name} and {report_path.name}", flush=True)
    else:
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
        report = [
            {"id": f"eval-{index:02d}", "prompt": text} for index, text in enumerate(kept[12:])
        ]
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
        "batch_size": GENERATE_BATCHES[0],
        "max_new_tokens": 1024,
        "conditions": conditions,
        "judge": {
            "feature": prompts["feature"],
            "config": str(JUDGE_YAML),
            "batch_size": 8,
        },
    }
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


_score_pool: ThreadPoolExecutor | None = None
_score_futures: list[tuple[Path, Path, Path, Future[None]]] = []
_score_lock = threading.Lock()


def _pipeline_argv(path: Path, output: Path, mode: str) -> list[str]:
    argv = [
        str(PY),
        "experiments/pipeline/run.py",
        mode,
        "--config",
        str(path),
        "--output",
        str(output),
    ]
    if os.environ.get("LAUNCH_JUDGE", "1") != "0":
        argv.append("--run-judge")
    return argv


def generate_config(path: Path, output: Path, gpu: str, log_path: Path) -> None:
    """Generate on one GPU. Scoring is a later process and does not keep the card."""
    for batch in GENERATE_BATCHES:
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("batch_size") != batch:
            plan["batch_size"] = batch
            path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        code = log_call(_pipeline_argv(path, output, "generate"), gpu, log_path)
        if code == 0:
            submit_score(path, output, log_path)
            return
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:].lower()
        if "out of memory" not in tail:
            raise RuntimeError(f"{path.name} failed, see {log_path}")
        print(f"{path.name} ran out of memory at batch {batch}", flush=True)
    raise RuntimeError(f"{path.name} ran out of memory at batch {GENERATE_BATCHES[-1]}")


def score_config(path: Path, output: Path, log_path: Path) -> None:
    code = log_call(_pipeline_argv(path, output, "score"), None, log_path)
    if code:
        raise RuntimeError(f"{path.name} score failed, see {log_path}")


def submit_score(path: Path, output: Path, log_path: Path) -> None:
    if _score_pool is None:
        raise RuntimeError("score pool is not started")
    with _score_lock:
        if any(queued == path and not future.done() for queued, _, _, future in _score_futures):
            return
    future = _score_pool.submit(score_config, path, output, log_path)
    with _score_lock:
        _score_futures.append((path, output, log_path, future))


def wait_scores() -> list[tuple[Path, Path, Path]]:
    with _score_lock:
        pending = list(_score_futures)
        _score_futures.clear()
    failed: list[tuple[Path, Path, Path]] = []
    for path, output, log_path, future in pending:
        try:
            future.result()
        except Exception as error:
            failed.append((path, output, log_path))
            print(f"FAILED score {path.name}: {error}", flush=True)
        else:
            print(f"scored {path.name}", flush=True)
    return failed


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


def run_batched(argv: list[str], batches: tuple[int, ...], gpu: str, log_path: Path) -> int:
    """Run ``argv`` once per batch. A CUDA OOM tries the next, smaller batch."""
    code = 1
    for batch in batches:
        code = log_call([*argv, "--batch-size", str(batch)], gpu, log_path)
        if code == 0:
            return 0
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:].lower()
        if "out of memory" not in tail:
            return code
        print(f"{log_path.name} ran out of memory at batch {batch}", flush=True)
    return code


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
        "--output",
        str(output),
    ]
    if concept.get("source"):
        argv += ["--source", concept["source"], "--target", concept["target"]]
    log = LOGS / f"{spec['name']}-{concept['name']}-direction.log"
    if run_batched(argv, EXTRACT_BATCHES, gpu, log) or not direction_ready(directory):
        raise RuntimeError(f"direction failed for {spec['name']} {concept['name']}")
    return directory


def gdn_job(concept: dict, spec: dict, prompts: dict, gpu: str) -> None:
    prompts = {**prompts, "feature": concept["feature"]}
    direction = extract_direction(concept, spec, gpu)
    conditions = steer_conditions(direction, spec["method"])
    output = RUNS / "pipeline" / f"{spec['name']}-{concept['name']}"
    for bench_name, bench in prompts["benchmarks"].items():
        config = PLANS / f"{spec['name']}-{concept['name']}-{bench_name}.json"
        if not config.exists():
            write_plan(config, spec, bench, conditions, prompts)
        generate_config(
            config, output, gpu, LOGS / f"{spec['name']}-{concept['name']}-{bench_name}.log"
        )


def gdn_clamp_conditions(direction: Path, method: str = "gdn_clamp") -> list[dict]:
    """Rank-1 clamp. Scale 1 matches the rank-1 additive gain."""
    rows = steer_conditions(direction, method)
    clamp = next(row for row in rows if row["name"] == "rank1")
    clamp["name"] = "clamp"
    clamp.pop("prompt_position", None)
    return [rows[0], clamp]


def gdn_clamp_job(concept: dict, spec: dict, prompts: dict, gpu: str) -> None:
    prompts = {**prompts, "feature": concept["feature"]}
    direction = extract_direction(concept, spec, gpu)
    tag = "mamba-clamp" if spec["method"] == "mamba_clamp" else "gdn-clamp"
    conditions = gdn_clamp_conditions(direction, spec["method"])
    output = RUNS / "pipeline" / f"{spec['name']}-{concept['name']}-{tag}"
    for bench_name, bench in prompts["benchmarks"].items():
        config = PLANS / f"{spec['name']}-{concept['name']}-{tag}-{bench_name}.json"
        if not config.exists():
            write_plan(config, spec, bench, conditions, prompts)
        generate_config(config, output, gpu, LOGS / f"{config.stem}.log")


def mamba_add_job(concept: dict, spec: dict, prompts: dict, gpu: str) -> None:
    """Rank-1, rank-2, and full additive Mamba. Scale 1 matches the full direction."""
    prompts = {**prompts, "feature": concept["feature"]}
    direction = extract_direction(concept, spec, gpu)
    conditions = steer_conditions(direction, "mamba_add")
    tag = "mamba-add"
    output = RUNS / "pipeline" / f"{spec['name']}-{concept['name']}-{tag}"
    for bench_name, bench in prompts["benchmarks"].items():
        config = PLANS / f"{spec['name']}-{concept['name']}-{tag}-{bench_name}.json"
        if not config.exists():
            write_plan(config, spec, bench, conditions, prompts)
        generate_config(config, output, gpu, LOGS / f"{config.stem}.log")


def residual_language_job(concept: dict, spec: dict, prompts: dict, gpu: str) -> None:
    """Additive and clamp residual steering at a fixed layer, on every token."""
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
            "--output",
            str(output),
        ]
        if run_batched(argv, EXTRACT_BATCHES, gpu, log) or not direction_ready(directory):
            raise RuntimeError(f"residual extract failed for {concept['name']}")
    if "layer" in concept:
        layer = int(concept["layer"])
    else:
        choice = output / "layer.json"
        if not choice.exists():
            questions = RUNS / "forgetting" / "questions.jsonl"
            code = log_call(
                [
                    str(PY),
                    "experiments/pipeline/residual.py",
                    "probe",
                    "--model",
                    spec["model"],
                    "--direction",
                    str(directory),
                    "--feature",
                    concept["feature"],
                    "--questions",
                    str(questions),
                    "--output",
                    str(output),
                ],
                gpu,
                log,
            )
            if code or not choice.exists():
                raise RuntimeError(f"layer probe failed for {concept['name']}")
        layer = int(json.loads(choice.read_text(encoding="utf-8"))["layer"])
        print(f"{concept['name']} residual layer {layer}", flush=True)
    conditions = [
        {"name": "baseline", "method": "baseline", "scales": [0]},
        {
            "name": "add",
            "method": "residual_add",
            "direction": str(directory),
            "layer": layer,
            "normalize": False,
            "scales": SCALES,
        },
        {
            "name": "clamp",
            "method": "residual_clamp",
            "direction": str(directory),
            "layer": layer,
            "normalize": False,
            "scales": SCALES,
        },
    ]
    result = RUNS / "pipeline" / f"{spec['name']}-{concept['name']}-residual"
    for bench_name, bench in prompts["benchmarks"].items():
        config = PLANS / f"{spec['name']}-{concept['name']}-residual-{bench_name}.json"
        if not config.exists():
            write_plan(config, spec, bench, conditions, prompts)
        generate_config(config, result, gpu, LOGS / f"{config.stem}.log")


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
        generate_config(config, result, gpu, LOGS / f"{config.stem}.log")


def main() -> None:
    global _score_pool
    _score_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="score")
    LOGS.mkdir(parents=True, exist_ok=True)
    pid_name = "falcon.pid" if len(sys.argv) > 1 and sys.argv[1] == "falcon" else "queue.pid"
    pid = LOGS / pid_name
    if pid.exists() and Path(f"/proc/{pid.read_text().strip()}").exists():
        raise SystemExit(f"queue already running as {pid.read_text().strip()}")
    pid.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "languages":
            _run_languages()
        elif len(sys.argv) > 1 and sys.argv[1] == "falcon":
            _run_falcon()
        else:
            _run_campaign()
    finally:
        _score_pool.shutdown(wait=True)


def _language_jobs(
    spec: dict,
    prompts: dict,
    concepts: tuple[dict, ...],
    *,
    require_direction: bool,
    additive: bool = False,
) -> list[Job]:
    jobs = []
    for concept in concepts:
        jobs.append(
            Job(
                f"{spec['name']}-{concept['name']}-residual",
                lambda gpu, c=concept: residual_language_job(c, spec, prompts, gpu),
            )
        )
        ready = direction_ready(
            RUNS / "directions" / f"{spec['name']}-{concept['name']}" / "direction"
        )
        if ready or not require_direction:
            jobs.append(
                Job(
                    f"{spec['name']}-{concept['name']}-clamp",
                    lambda gpu, c=concept: gdn_clamp_job(c, spec, prompts, gpu),
                )
            )
        if additive:
            jobs.append(
                Job(
                    f"{spec['name']}-{concept['name']}-mamba-add",
                    lambda gpu, c=concept: mamba_add_job(c, spec, prompts, gpu),
                )
            )
    return jobs


def _finish_language_plots(prefix: str, kinds: tuple[str, ...]) -> None:
    from figures import write_tradeoff

    for kind in kinds:
        path = write_tradeoff(
            RUNS / "pipeline", kind, RUNS / f"{prefix}-{kind}-tradeoff.html", prefix=prefix
        )
        print(f"wrote {path}", flush=True)


def _run_languages() -> None:
    """Qwen language sweeps: residual add and clamp, plus GDN clamp."""
    os.environ["LAUNCH_JUDGE"] = "0"
    spec = {"name": "qwen", "model": "Qwen/Qwen3.5-9B", "method": "gdn_clamp"}
    prompts = prepare(models=(spec,))
    print("language data is ready", flush=True)
    run_phase(_language_jobs(spec, prompts, LANGUAGE_LAYERS, require_direction=True))
    _finish_language_plots("qwen", ("residual", "gdn-clamp"))
    print(f"finished language sweeps under {RUNS / 'pipeline'}", flush=True)


def _run_falcon() -> None:
    """Falcon language sweeps: residual, Mamba clamp, and additive Mamba at three ranks."""
    global GENERATE_BATCHES, EXTRACT_BATCHES
    GENERATE_BATCHES = FALCON_GENERATE_BATCHES
    EXTRACT_BATCHES = FALCON_EXTRACT_BATCHES
    os.environ["LAUNCH_JUDGE"] = "0"
    os.environ.setdefault("GPU_FREE_MIB", "20000")
    spec = {
        "name": "falcon",
        "model": "tiiuae/Falcon-H1-7B-Instruct",
        "method": "mamba_clamp",
    }
    languages = tuple(
        {key: value for key, value in concept.items() if key != "layer"}
        for concept in LANGUAGE_LAYERS
    )
    prompts = prepare(models=(spec,))
    print("falcon data is ready", flush=True)
    run_phase(_language_jobs(spec, prompts, languages, require_direction=False, additive=True))
    _finish_language_plots("falcon", ("residual", "mamba-clamp", "mamba-add"))
    print(f"finished falcon sweeps under {RUNS / 'pipeline'}", flush=True)


def _run_campaign() -> None:
    prompts = prepare()
    print("data is ready, waiting for a free GPU", flush=True)
    skip_gdn = os.environ.get("LAUNCH_FROM") == "residual"
    for spec in MODELS:
        if skip_gdn:
            skip_gdn = False
            print(f"skipping finished {spec['name']} GDN", flush=True)
        else:
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
