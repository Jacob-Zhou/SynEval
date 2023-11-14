#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --ntasks-per-node=1

set -o nounset
set -o errexit
set -o pipefail

{
    export TRANSFORMERS_OFFLINE=1
    . scripts/set_environment.sh
    for i in {0..2}; do
        python -u test.py --model-name random --suite v23 --force --exemplar-type syntactic-knowledge --seed $i --n-exemplar 0
    done
}