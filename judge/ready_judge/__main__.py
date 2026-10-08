import argparse
import getpass
import json
from .core import evaluate


def main():
    parser = argparse.ArgumentParser(description='Blind variable-scale Judge release candidate')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resources', help='Explicit historical or candidate resources; default is multilingual-complexity6.3.0-review1')
    parser.add_argument('--run', action='store_true', help='Explicitly send paid API requests')
    parser.add_argument('--prompt-key', action='store_true', help='Hidden key input; never saved')
    args = parser.parse_args()
    if args.prompt_key and not args.run:
        parser.error('--prompt-key requires --run')
    secret = getpass.getpass('OpenRouter key (not saved): ') if args.prompt_key else None
    result = evaluate(args.input, args.output, args.features, args.run, args.workers, secret, args.resources)
    print(json.dumps(result))
    if args.run and not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
