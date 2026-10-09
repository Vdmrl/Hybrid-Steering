"""Paired GDN/residual concept comparison; reuse the project's decoder and math."""
import argparse
import csv
import hashlib
import json
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path
from hybrid_steering.paths import ROOT as PROJECT_ROOT

import torch
from hybrid_steering import language_benchmark as bench
from hybrid_steering import language_search as search
from hybrid_steering import runner as run
from hybrid_steering.residual import digest, residual_hook
from hybrid_steering.state import analyze, reconstruct, sparsify_heads
from hybrid_steering.architecture import decoder_layers, recurrent_family
from hybrid_steering.extract import conditioned_prompt


def batch_size(config: dict) -> int:
    """Execution-only override; recorded separately from the scientific config."""
    size = int(os.environ.get('GDN_RUNTIME_BATCH_SIZE', config['batch_size']))
    assert size > 0
    return size


def generate(model, tokenizer, rows: list, limit: int, direction=None, strength: float = 0.) -> list:
    return run.generate_batch(model, tokenizer, [r['prompt'] for r in rows], direction,
                              None, strength, 'once', 'cuda', limit,
                              system=None, enable_thinking=False)


@torch.inference_mode()
def prepare(model, tokenizer, train: list, config: dict, root: Path) -> dict:
    artifact_path = root / 'directions.pt'
    pair_path = root / 'train-answers.jsonl'
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
        assert artifact['pairs_sha256'] == digest(pair_path)
        if 'extraction_config' in artifact:
            assert artifact['extraction_config'] == config
        return artifact
    pairs = bench.read_jsonl(pair_path)
    assert len({r['id'] for r in pairs}) == len(pairs)
    done = {r['id'] for r in pairs}
    todo = [r for r in train if r['id'] not in done]
    for start in range(0, len(todo), batch_size(config)):
        rows = todo[start:start + batch_size(config)]
        source = [{**r, 'prompt': conditioned_prompt(r['prompt'], config.get('source_instruction'))}
                  for r in rows]
        conditioned = [{**r, 'prompt': conditioned_prompt(r['prompt'], config['instruction'])}
                       for r in rows]
        outputs = []
        for batch_rows in (source, conditioned):
            outputs.append(bench.adaptive_batches(batch_rows, batch_size(config),
                lambda batch: generate(model, tokenizer, batch, config['pair_max_new_tokens'])))
        saved = [{**r, 'neutral_response': neg, 'positive_response': pos}
                 for r, neg, pos in zip(rows, *outputs)]
        bench.append_jsonl(pair_path, saved)
        pairs += saved
        print(f'prepare pairs={len(pairs)}/{len(train)}', flush=True)
    assert len(pairs) == len(train)
    assert all(r['neutral_response'].strip() and r['positive_response'].strip() for r in pairs)
    blocks = decoder_layers(model)
    residual = {name: {layer: [] for layer in config['residual_layers']} for name in ('neutral', 'positive')}
    states = {'neutral': {}, 'positive': {}}
    # One centroid per answer prefix; instruction direction uses input boundary.
    for name in residual:
        extraction_batch = config.get('extraction_batch_size', 4)
        for start in range(0, len(pairs), extraction_batch):
            rows = pairs[start:start + extraction_batch]
            instruction = config['instruction'] if name == 'positive' else config.get('source_instruction')
            texts = [run.chat(tokenizer, conditioned_prompt(r['prompt'], instruction),
                              system=None, enable_thinking=False) for r in rows]
            if config.get('residual_representation') == 'response_prefix':
                sequences = []
                for row in rows:
                    prompt = run.chat(tokenizer, row['prompt'], system=None, enable_thinking=False)
                    sequences.append(tokenizer.encode(prompt, add_special_tokens=False) +
                        tokenizer.encode(row[name + '_response'], add_special_tokens=False)[:config['prefix_tokens']])
                batch = tokenizer.pad({'input_ids': sequences}, padding=True, return_tensors='pt').to('cuda')
            else:
                batch = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
            handles = []
            for layer in config['residual_layers']:
                def capture(module, args, output, layer=layer):
                    hidden = output[0] if isinstance(output, tuple) else output
                    residual[name][layer].append(hidden[:, -1].float().cpu())
                handles.append(blocks[layer].register_forward_hook(capture))
            try:
                mask = batch.attention_mask
                positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
                model(**batch, position_ids=positions, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            if config.get('extract_gdn', True):
                # No tone instruction during GDN extraction, only model-produced answer prefix.
                texts = [run.chat(tokenizer, r['prompt'], system=None, enable_thinking=False) +
                         search.truncate(tokenizer, r[name + '_response'], config['prefix_tokens']) for r in rows]
                for layer, value in run.recurrent_states_batch(model, tokenizer, texts, 'cuda').items():
                    states[name][layer] = states[name].get(layer, torch.zeros_like(value[0])) + value.sum(0)
            print(f'extract {name}={min(start+extraction_batch,len(pairs))}/{len(pairs)}', flush=True)
    direction = {layer: (states['positive'][layer] - states['neutral'][layer])/len(pairs)
                 for layer in states['positive']}
    variants = {'gdn_full': direction}
    requested = config.get('gdn_methods')
    ranks = (1, 5) if requested is None else [rank for rank in (1, 5)
        if f'gdn_rank{rank}' in requested or (rank == 1 and 'gdn_clamp_rank1' in requested)
        or (rank == 5 and config.get('head_percentages'))]
    for rank in ranks:
        _, factors = analyze(direction, 'cpu', rank=rank)
        variants[f'gdn_rank{rank}'] = reconstruct(factors)
        if rank == 1:
            variants['clamp_factors'] = factors
    sparsity = {}
    for percent in config.get('head_percentages', []):
        name = f'gdn_rank5_heads{percent}'
        variants[name], sparsity[name] = sparsify_heads(variants['gdn_rank5'], percent / 100)
    units, targets, neutral_targets, residual_directions = {}, {}, {}, {}
    for layer in config['residual_layers']:
        positive = torch.cat(residual['positive'][layer])
        neutral = torch.cat(residual['neutral'][layer])
        if config.get('residual_representation') == 'response_prefix':
            positive = positive[:config['residual_train_count']]
            neutral = neutral[:config['residual_train_count']]
        difference = positive.mean(0) - neutral.mean(0)
        assert torch.isfinite(difference).all() and difference.norm() > 0
        units[layer] = difference / difference.norm()
        targets[layer] = (positive @ units[layer]).mean()
        neutral_targets[layer] = (neutral @ units[layer]).mean()
        if config.get('residual_representation') == 'response_prefix':
            residual_directions[layer] = units[layer] * neutral.norm(dim=-1).mean()
    artifact = {**variants, 'sparsity': sparsity,
                'units': units, 'targets': targets, 'neutral_targets': neutral_targets,
                'pairs_sha256': digest(pair_path),
                'residual_directions': residual_directions, 'extraction_config': config,
                'state_family': recurrent_family(model),
                'layer_types': {l: getattr(blocks[l], 'block_type', recurrent_family(model)) for l in config['residual_layers']}}
    torch.save(artifact, artifact_path)
    return artifact


def jobs(config: dict, benchmark: dict | None = None) -> list:
    if benchmark:
        cases = [tuple(case) for case in benchmark['cases']]
        for field in ('residual_strengths','gdn_strengths'):
            if field not in benchmark:continue
            # Evaluation amplitudes do not change the extracted direction.
            import math
            strengths = benchmark[field]
            assert strengths and len(strengths) == len(set(strengths))
            assert all(type(c) in (int,float) and math.isfinite(c) and c > 0 for c in strengths)
            config = {**config, field:strengths}
        if benchmark.get('normalize_rank1_to_full'):
            assert all(0 < c <= 3 for m,l,c in cases if m=='gdn_clamp_rank1')
        allowed = set(jobs(config))
        assert cases and len(cases) == len(set(cases)) and set(cases) <= allowed
        return cases
    methods = config.get('gdn_methods', ['gdn_full', 'gdn_rank5'])
    return [('baseline', -1, 0.)] + [(method, -1, c) for method in methods
        for c in config['gdn_strengths']] + [('residual', layer, c) for layer in config['residual_layers']
        for c in config['residual_strengths']]


def generate_condition(model, tokenizer, rows: list, artifact: dict, method: str,
                       layer: int, strength: float, limit: int) -> list:
    if method == 'gdn_clamp_rank1':
        return run.generate_batch(model, tokenizer, [r['prompt'] for r in rows], None,
            artifact['clamp_factors'], strength, 'clamp', 'cuda', limit,
            system=None, enable_thinking=False)
    hook = nullcontext()
    if method in {'residual', 'residual_clamp'}:
        unit = artifact['units'][layer]
        # The stored scalar gap is <mean(target) - mean(neutral), unit> = ||v||.
        # Thus v = (target_projection - neutral_projection) * unit.
        gap = artifact['targets'][layer] - artifact['neutral_targets'][layer]
        if method == 'residual':
            hook = residual_hook(decoder_layers(model)[layer], strength * gap * unit, 'every')
        else:
            hook = residual_hook(decoder_layers(model)[layer], unit,
                                 clamp_target=strength * gap)
    with hook:
        return generate(model, tokenizer, rows, limit, artifact.get(method),
                        strength if method.startswith('gdn_') else 0.)


def validate_answers(rows: list, evaluation: list, cases: list) -> None:
    prompts = {r['id']: r['prompt'] for r in evaluation}
    expected = {(*case, key) for case in cases for key in prompts}
    actual = {(r['method'], r['layer'], r['strength'], r['id']) for r in rows}
    assert len(prompts) == len(evaluation)
    assert len(rows) == len(actual) == len(expected) and actual == expected
    assert all(r['prompt'] == prompts[r['id']] and not r['thinking'] for r in rows)


def score_ifeval(root: Path, config: dict, benchmark: dict) -> None:
    """Score saved answers only, using the shared pinned official evaluator."""
    evaluation = [{**r, 'id': r['key']} for r in bench.read_jsonl(root / benchmark['input'])]
    assert len(evaluation) == benchmark['expected_n']
    assert 0 < len(evaluation) <= 541
    rows = bench.read_jsonl(root / 'answers.jsonl')
    cases = jobs(config, benchmark)
    validate_answers(rows, evaluation, cases)
    metrics, scored = [], []
    for method, layer, strength in cases:
        selected = [r for r in rows if (r['method'], r['layer'], r['strength']) == (method, layer, strength)]
        quality, flags = bench.ifeval_scores(selected)
        if config.get('language_code'):
            labels = {r['prompt']: search.language(r['response']) for r in selected}
            quality['language_rate'] = sum(label == config['language_code']
                                           for label in labels.values()) / len(selected)
            flags = {prompt: {**value, 'detected_language': labels[prompt]}
                     for prompt, value in flags.items()}
        metrics.append({'method': method, 'layer': layer, 'strength': strength,
                        'n': len(selected), **quality})
        scored += [{**r, **flags[r['prompt']]} for r in selected]
    bench.write_jsonl(root / 'scored.jsonl', scored)
    with (root / 'metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    print('IFEVAL SCORED: concept intensity still requires the blind judge', flush=True)


@torch.inference_mode()
def main(config_path: Path, prepare_only: bool = False, smoke: bool = False,
         benchmark_path: Path | None = None, score_only: bool = False) -> None:
    config = json.loads(config_path.read_text())
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path else None
    generation = benchmark or config
    assert generation['enable_thinking'] is False and generation['system'] is None
    root = bench.ROOT
    root.mkdir(parents=True, exist_ok=True)
    if score_only:
        assert benchmark is not None
        manifest = json.loads((root / 'manifest.json').read_text())
        assert manifest['identity']['config'] == config
        assert manifest['identity']['benchmark'] == benchmark
        assert manifest['identity']['dataset_sha256'] == digest(root / benchmark['input'])
        score_ifeval(root, config, benchmark)
        return
    source = root / 'inputs/pairs.jsonl'
    rows = [{'id': r['id'], 'prompt': r['prompt']} for r in bench.read_jsonl(source)]
    train, evaluation = rows[:config['train_count']], rows[config['train_count']:]
    if config.get('eval_prompt_config'):
        extra = [{'id': i, 'prompt': text} for i, text in enumerate(
            json.loads(Path(config['eval_prompt_config']).read_text())['prompts'])]
    else:
        extra = search.prompts(json.loads(Path(config['neutral_prompt_config']).read_text()))
    evaluation += [{**r, 'id': f'neutral-{r["id"]}'} for r in extra]
    if config.get('eval_limit'):
        evaluation = sorted(evaluation, key=lambda r: hashlib.sha256(r['prompt'].encode()).hexdigest())[:config['eval_limit']]
    source_manifest = None
    if benchmark:
        # Reuse the completed screen's model-generated pairs and directions.
        source_manifest = json.loads((root / 'source-manifest.json').read_text())
        assert source_manifest['identity']['config'] == config
        assert source_manifest['identity']['source_sha256'] == digest(source)
        assert (root / 'directions.pt').is_file() and (root / 'train-answers.jsonl').is_file()
        evaluation = [{**r, 'id': r['key']} for r in bench.read_jsonl(root / benchmark['input'])]
        assert len(evaluation) == benchmark['expected_n']
        assert 0 < len(evaluation) <= 541
    assert not {r['prompt'] for r in train} & {r['prompt'] for r in evaluation}
    assert len({r['id'] for r in evaluation}) == len(evaluation)
    code = {p: digest(PROJECT_ROOT / p) for p in (
        "src/hybrid_steering/comparison.py", "src/hybrid_steering/residual.py",
        "src/hybrid_steering/state.py", "src/hybrid_steering/runner.py",
        "src/hybrid_steering/language_benchmark.py", "src/hybrid_steering/language_search.py")}
    if config.get('placement', {}).get('transfer_via_cpu'):
        code["src/hybrid_steering/cpu_transfer.py"] = digest(PROJECT_ROOT / "src/hybrid_steering/cpu_transfer.py")
    identity = {'config': config, 'source_sha256': digest(source), 'code': code,
                'train': train, 'evaluation': evaluation}
    if benchmark:
        identity.update(benchmark=benchmark, dataset_sha256=digest(root / benchmark['input']),
                        direction_sha256=digest(root / 'directions.pt'),
                        source_manifest_sha256=digest(root / 'source-manifest.json'))
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['identity'] == identity, 'Changed resume protocol'
    model, tokenizer = bench.load_model(config['model'], config.get('placement'))
    if source_manifest:
        assert getattr(model.config, '_commit_hash', None) == source_manifest['model_revision']
    manifest_path.write_text(json.dumps({'identity': identity,
        'model_revision': getattr(model.config, '_commit_hash', None),
        'tokenizer_revision': tokenizer.init_kwargs.get('_commit_hash'),
        'host': socket.gethostname(), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'data_root': str(root), 'gpu': torch.cuda.get_device_name(),
        'device_map': getattr(model, 'hf_device_map', None),
        'runtime_batch_size': batch_size(generation),
        'evaluator_sha256': {name: digest(root / 'cache/google_research/instruction_following_eval' / name)
                             for name in bench.IFEVAL_FILES} if benchmark else None}, indent=2))
    artifact = prepare(model, tokenizer, train, config, root)
    if benchmark and benchmark.get('normalize_rank1_to_full'):
        from hybrid_steering.state import match_head_norm
        normalized, report = match_head_norm(artifact['gdn_rank1'],artifact['gdn_full'])
        assert all(r['capped_heads']==0 for r in report.values()), 'Normalization gain cap reached'
        _, factors = analyze(normalized,'cpu',rank=1)
        for layer, values in factors.items():
            device = next(decoder_layers(model)[layer].parameters()).device
            factors[layer] = tuple(v.to(device=device,dtype=torch.bfloat16) for v in values)
        artifact = {**artifact,'clamp_factors':factors}
        (root/'normalization.json').write_text(json.dumps(report,indent=2))
    if artifact.get('reused_source'):
        assert getattr(model.config, '_commit_hash', None) == artifact['reused_source']['model_revision']
    if config.get('residual_representation') == 'response_prefix':
        expected_layers = [i for i, kind in enumerate(model.config.text_config.layer_types)
                           if kind == 'linear_attention']
        if config.get('extract_gdn', True):
            assert sorted(artifact['gdn_full']) == expected_layers
        # Avoid CPU->GPU transfers of clamp factors at every generated token.
        from torch.utils._pytree import tree_map
        if not config.get('placement'):
            artifact = tree_map(lambda value: value.cuda() if isinstance(value, torch.Tensor) else value, artifact)
    if smoke:
        if config.get('residual_representation') == 'response_prefix':
            saved = []
            for method, layer, strength in jobs(config):
                answers = bench.adaptive_batches(evaluation[:4], min(4, batch_size(config)), lambda batch:
                    generate_condition(model, tokenizer, batch, artifact, method, layer,
                                       strength, config.get('smoke_max_new_tokens', 256)))
                assert len(answers) == len(evaluation[:4])
                if method == 'baseline':
                    assert all(answer.strip() for answer in answers)
                saved += [{**row, 'method': method, 'layer': layer, 'strength': strength,
                           'response': answer, 'thinking': False}
                          for row, answer in zip(evaluation[:4], answers)]
                print(f'SMOKE {method} layer={layer} c={strength} n={len(answers)}', flush=True)
            bench.write_jsonl(root/'smoke-answers.jsonl', saved)
            print('SMOKE COMPLETE: mechanical check, not concept validation', flush=True)
            return
        checks = [('baseline', -1, 0), ('gdn_full', -1, 3), ('gdn_rank5', -1, 3),
                  ('gdn_rank5_heads25', -1, 3), ('residual', 27, 1)]
        saved = []
        for method, layer, strength in checks:
            answers = generate_condition(model, tokenizer, evaluation[:2], artifact, method, layer, strength, 64)
            assert len(answers) == 2 and all(answer.strip() for answer in answers)
            saved += [{'method': method, 'response': answer} for answer in answers]
        bench.write_jsonl(root/'smoke-answers.jsonl', saved)
        print('SMOKE PASSED: real baseline/GDN/rank5/sparse/residual generation', flush=True)
        return
    if prepare_only:
        print('PREPARED: directions validated; no evaluation requested', flush=True)
        return
    artifact_sha = digest(root / 'directions.pt')
    path = root / 'answers.jsonl'
    old = bench.read_jsonl(path)
    done = {(r['method'], r['layer'], r['strength'], r['id']) for r in old}
    assert len(done) == len(old)
    for method, layer, strength in jobs(config, benchmark):
        todo = [r for r in evaluation if (method, layer, strength, r['id']) not in done]
        for start in range(0, len(todo), batch_size(generation)):
            rows = todo[start:start+batch_size(generation)]
            began = time.monotonic()
            def call(batch):
                return generate_condition(model, tokenizer, batch, artifact, method, layer,
                                          strength, generation['max_new_tokens'])
            answers = bench.adaptive_batches(rows, batch_size(generation), call)
            saved = [{**r, 'concept': config['concept'], 'method': method, 'layer': layer,
                      'strength': strength, 'response': answer, 'thinking': False,
                      'model': config['model'], 'direction_sha256': artifact_sha}
                     for r, answer in zip(rows, answers)]
            bench.append_jsonl(path, saved)
            print(f'{method} layer={layer} c={strength} n={len(rows)} seconds={time.monotonic()-began:.1f}', flush=True)
    all_rows = bench.read_jsonl(path)
    validate_answers(all_rows, evaluation, jobs(config, benchmark))
    # Blind IDs, with method mapping kept separate for a later judge.
    blind, mapping = [], []
    for row in all_rows:
        key = json.dumps([row['method'],row['layer'],row['strength'],row['id']])
        uid = hashlib.sha256(key.encode()).hexdigest()[:20]
        blind.append({'id': uid, 'prompt': row['prompt'], 'response': row['response']})
        mapping.append({'id': uid, 'method': row['method'], 'layer': row['layer'],
                        'strength': row['strength'], 'prompt_id': row['id']})
    bench.write_jsonl(root/'judge-input.jsonl', sorted(blind, key=lambda r:r['id']))
    bench.write_jsonl(root/'judge-key.jsonl', mapping)
    print(f'COMPLETE: {len(all_rows)} answers; judge not run', flush=True)
    if benchmark and benchmark.get('score_ifeval', True):
        score_ifeval(root, config, benchmark)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path("data/legacy-config/configs/4b/enthusiasm_comparison.json"))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    main(args.config, args.prepare_only, args.smoke, args.benchmark, args.score_only)
