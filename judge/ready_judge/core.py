"""Variable-scale blind scoring, strict provenance and explicit paid execution."""
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent / 'resources'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def normalized(score, maximum):
    if type(score) is not int or type(maximum) is not int or maximum < 1 or not 0 <= score <= maximum:
        raise ValueError('Score outside configured scale')
    return 100.0 * score / maximum


def configuration(root=ROOT):
    # JSON syntax is a valid YAML subset; no YAML/runtime dependency is needed.
    features = json.loads((root / 'concepts/features.yaml').read_text(encoding='utf-8'))
    config = json.loads((root / 'config/judge.json').read_text(encoding='utf-8'))
    if config['base_url'] != 'https://openrouter.ai/api/v1' or config['reasoning_enabled']:
        raise ValueError('Unsupported endpoint or reasoning mode')
    prompts = {}
    for name, feature in features['features'].items():
        maximum = feature['maximum']
        if type(maximum) is not int or not 1 <= maximum <= 9:
            raise ValueError('Scale must have single-digit positive maximum')
        if set(feature['anchors']) != {str(i) for i in range(maximum + 1)}:
            raise ValueError('Missing scale anchor')
        if not 0 <= feature['success_threshold'] <= maximum:
            raise ValueError('Invalid success threshold')
        template = (root / 'prompts' / feature['prompt']).read_text(encoding='utf-8')
        if feature.get('guide'):
            guide = (root / 'prompts' / feature['guide']).read_text(encoding='utf-8').strip()
        else:
            guide = '\n'.join([
                'Target: ' + feature['target'],
                'Definition: ' + feature['definition'],
                'Exclusions:\n' + '\n'.join('- ' + x for x in feature['exclusions']),
                'Anchors:\n' + '\n'.join(f'{k}: {v}' for k, v in feature['anchors'].items()),
            ])
        prompts[name] = template.replace('{{concept}}', guide).replace('{{maximum}}', str(maximum))
    return config, features, prompts


def parse_choice(choice, maximum):
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Incomplete score')
    digit = choice.get('message', {}).get('content')
    allowed = {str(i) for i in range(maximum + 1)}
    if digit not in allowed:
        raise ValueError('Response must be one valid ASCII score digit')
    score = int(digit)
    result = {'raw_score': score, 'scale_min': 0, 'scale_max': maximum,
              'normalized_score_pct': normalized(score, maximum),
              'score_distribution': {'probabilities': None, 'expected_raw_score': None,
                                     'expected_normalized_score_pct': None,
                                     'valid_token_mass': None, 'complete': False,
                                     'observed_label_logprobs': {},
                                     'observed_label_probabilities': {},
                                     'available_label_distribution': None,
                                     'available_expected_normalized_score_pct': None,
                                     'missing_labels': sorted(allowed),
                                     'source': 'missing_or_incomplete_token_logprobs'}}
    parts = (choice.get('logprobs') or {}).get('content')
    if not isinstance(parts, list) or not parts or parts[0].get('token') != digit:
        return result
    alternatives = parts[0].get('top_logprobs')
    if not isinstance(alternatives, list):
        return result
    values = {}
    for item in alternatives:
        token, value = item.get('token'), item.get('logprob')
        if token not in allowed:
            continue
        if token in values or type(value) not in (int, float) or not math.isfinite(value) or value > 0:
            raise ValueError('Invalid or duplicate label logprob')
        values[token] = value
    # Use every observed numeric label, without pretending omitted labels have zero mass.
    observed = {k: math.exp(v) for k, v in values.items()}
    distribution = result['score_distribution']
    distribution['observed_label_logprobs'] = values
    distribution['observed_label_probabilities'] = observed
    distribution['missing_labels'] = sorted(allowed - set(values))
    if not values:
        return result
    offset = max(values.values())
    weights = {k: math.exp(v - offset) for k, v in values.items()}
    total = sum(weights.values())
    probabilities = {k: v / total for k, v in weights.items()}
    distribution['available_label_distribution'] = probabilities
    distribution['available_expected_normalized_score_pct'] = 100 * sum(int(k) * v for k, v in probabilities.items()) / maximum
    if set(values) != allowed:
        distribution['source'] = 'partial_token_logprobs_available_labels_only'
        return result
    if probabilities[digit] + 1e-12 < max(probabilities.values()):
        raise ValueError('Score conflicts with returned label logprobs')
    mass = sum(math.exp(v) for v in values.values())
    if not 0 < mass <= 1.00001:
        raise ValueError('Invalid label probability mass')
    expected = sum(int(k) * v for k, v in probabilities.items())
    result['score_distribution'] = {
        'probabilities': probabilities, 'expected_raw_score': expected,
        'expected_normalized_score_pct': 100 * expected / maximum,
        'valid_token_mass': min(mass, 1.0), 'complete': True,
        'observed_label_logprobs': values, 'observed_label_probabilities': observed,
        'available_label_distribution': probabilities,
        'available_expected_normalized_score_pct': 100 * expected / maximum,
        'missing_labels': [],
        'source': 'token_logprobs_conditional_on_configured_labels',
    }
    return result


def input_rows(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if 'answers' in item:
            rows.extend({'prompt_id': item['prompt_id'], 'answer_id': a['answer_id'],
                         'scenario': item['scenario'], 'text': a['text']} for a in item['answers'])
        else:
            rows.append(item)
    identities = set()
    for row in rows:
        for key in ('prompt_id', 'answer_id', 'scenario', 'text'):
            if not isinstance(row.get(key), str) or (key != 'text' and not row[key]):
                raise ValueError('Input needs string prompt_id/answer_id/scenario/text')
        identity = row['prompt_id'], row['answer_id']
        if identity in identities:
            raise ValueError('Duplicate prompt_id/answer_id')
        identities.add(identity)
    if not rows:
        raise ValueError('Empty evaluation input')
    return rows


def payload(task, config, prompt, maximum):
    return {
        'model': config['model'], 'temperature': config['temperature'],
        'max_tokens': config['max_tokens'], 'seed': config['seed'],
        'reasoning': {'enabled': False}, 'logprobs': True,
        'top_logprobs': config['top_logprobs'],
        'provider': {'only': config['provider_only'], 'allow_fallbacks': config['allow_fallbacks'],
                     'require_parameters': True},
        'messages': [
            {'role': 'system', 'content': prompt + f'\nReturn exactly one ASCII digit 0-{maximum}, and nothing else.'},
            {'role': 'user', 'content': json.dumps({'prompt': task['scenario'], 'response': task['text']}, ensure_ascii=False)},
        ],
    }


def request_score(task, config, prompt, feature, secret, output):
    body = json.dumps(payload(task, config, prompt, feature['maximum'])).encode('utf-8')
    proxy = os.environ.get('OPENROUTER_PROXY', '').strip()
    opener = build_opener(ProxyHandler({'https': proxy} if proxy else {}))
    for attempt in range(config['max_attempts']):
        try:
            request = Request(config['base_url'] + '/chat/completions', data=body,
                              headers={'Authorization': 'Bearer ' + secret, 'Content-Type': 'application/json'})
            with opener.open(request, timeout=config['timeout_seconds']) as response:
                raw = json.load(response)
            saved = {k: raw.get(k) for k in ('id', 'model', 'provider', 'usage', 'choices')}
            raw_path = output / 'raw' / f'{task["task_id"]}-{attempt}-{time.time_ns()}.json'
            write(raw_path, saved)
            if raw.get('model') != config['model'] or raw.get('provider') not in config['provider_only']:
                raise ValueError('Unexpected model/provider')
            details = (raw.get('usage') or {}).get('completion_tokens_details') or {}
            message = raw['choices'][0].get('message', {})
            if details.get('reasoning_tokens', 0) or message.get('reasoning') or message.get('reasoning_details'):
                raise ValueError('Provider emitted disabled reasoning')
            label = parse_choice(raw['choices'][0], feature['maximum'])
            label['success'] = label['raw_score'] >= feature['success_threshold']
            return {k: task[k] for k in ('task_id', 'prompt_id', 'answer_id', 'feature')} | label | {
                'requested_model': config['model'], 'returned_model': raw['model'],
                'provider': raw['provider'], 'usage': raw.get('usage'), 'request_id': raw.get('id'),
                'raw_file': str(raw_path.relative_to(output)),
                'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'attempts': attempt + 1,
            }
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f'HTTP {error.code}; body withheld') from None
        except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
            pass
        if attempt + 1 < config['max_attempts']:
            time.sleep(2 ** attempt)
    raise RuntimeError('Request failed; score remains missing')


def evaluate(input_path, output, selected, run=False, workers=4, secret=None):
    if not 1 <= workers <= 16:
        raise ValueError('workers must be1-16')
    config, rubric, prompts = configuration()
    if len(set(selected)) != len(selected) or any(f not in rubric['features'] for f in selected):
        raise ValueError('Unknown or duplicate feature')
    rows = input_rows(input_path)
    identity = {'schema_version': '5.0.0-rc1', 'config': config, 'rubric': rubric,
                'prompt_sha256': {f: fingerprint(prompts[f]) for f in selected},
                'features': selected, 'seed': config['seed'], 'input_sha256': fingerprint(rows),
                'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    tasks = []
    for row in rows:
        for name in selected:
            task = {k: row[k] for k in ('prompt_id', 'answer_id', 'scenario', 'text')}
            task['feature'] = name
            task['task_id'] = fingerprint({'protocol': identity, 'task': task})
            tasks.append(task)
    random.Random(config['seed']).shuffle(tasks)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'evaluation.lock'
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Output is locked; inspect existing process before removing stale lock') from None
    os.close(fd)
    try:
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))['identity'] != identity:
            raise ValueError('Resume rejected: input, rubric, prompt, code or configuration changed')
        write(manifest_path, {'identity': identity, 'task_order': [t['task_id'] for t in tasks]})
        labels_path = output / 'scores.jsonl'
        previous = [json.loads(l) for l in labels_path.read_text(encoding='utf-8').splitlines()] if labels_path.exists() else []
        known = {t['task_id']: t for t in tasks}
        done = set()
        for row in previous:
            if row['task_id'] in done or row['task_id'] not in known:
                raise ValueError('Invalid or duplicate resumed task')
            feature = rubric['features'][row['feature']]
            if row['normalized_score_pct'] != normalized(row['raw_score'], feature['maximum']):
                raise ValueError('Invalid resumed score')
            done.add(row['task_id'])
        pending = [t for t in tasks if t['task_id'] not in done]
        if run and pending:
            secret = secret or os.environ.get('OPENROUTER_API_KEY', '').strip()
            if not secret or '\n' in secret:
                raise ValueError('Set OPENROUTER_API_KEY or use --prompt-key')
            def perform(task):
                try:
                    return request_score(task, config, prompts[task['feature']], rubric['features'][task['feature']], secret, output)
                except Exception as error:
                    return {'task_id': task['task_id'], 'error_type': type(error).__name__}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                with labels_path.open('a', encoding='utf-8') as scores, (output / 'errors.jsonl').open('a', encoding='utf-8') as errors:
                    for n, result in enumerate(pool.map(perform, pending), 1):
                        stream = errors if 'error_type' in result else scores
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                        stream.flush()
                        if 'raw_score' in result:
                            done.add(result['task_id'])
                        if n % 20 == 0:
                            print(json.dumps({'processed': n, 'pending_at_start': len(pending)}), flush=True)
        progress = {'expected': len(tasks), 'completed': len(done), 'missing': len(tasks) - len(done),
                    'complete': len(tasks) == len(done), 'paid_requests_enabled': run}
        write(output / 'progress.json', progress)
        return progress
    finally:
        lock_path.unlink(missing_ok=True)
