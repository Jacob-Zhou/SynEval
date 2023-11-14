#!/bin/bash

set -o nounset
set -o errexit
set -o pipefail

{
    export TRANSFORMERS_OFFLINE=1
    . scripts/set_environment.sh
    python -u test.py --model-name gpt-3.5-turbo-0613 --suite v23 --exemplar-type syntactic-knowledge --seed 0 --n-exemplar 0 --n-workers 8 --sample-size 1
    python -u test.py --model-name gpt-4-0613 --suite v23 --exemplar-type syntactic-knowledge --seed 0 --n-exemplar 0 --sample-size 1

    python -u test.py --model-name gpt-3.5-turbo-0613 --suite v23 --exemplar-type syntactic-knowledge --seed 0 --sample-size 5 --n-workers 8
    python -u test.py --model-name gpt-4-0613 --suite v23 --exemplar-type syntactic-knowledge --seed 0 --sample-size 5
}