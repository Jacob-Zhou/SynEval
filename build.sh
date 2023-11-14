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
    . scripts/set_environment.sh
    python tree_info_extractor.py --output extracted
    python question_generator.py --output generated
    python split_qa.py
}