#!/bin/bash
# Path: run_baichuan_revision.sh
# This file generates temperary scripts and submits them to the cluster.

set -o nounset
set -o errexit
set -o pipefail

{

    # key variables
    suite="v23"
    sample_size="5"

    base_model="baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints"
    # models that fit in single A100
    revisions=(
        "train_00220B"
        "train_00440B"
        "train_00660B"
        "train_00880B"
        "train_01100B"
        "train_01320B"
        "train_01540B"
        "train_01760B"
        "train_01980B"
        "train_02200B"
        "train_02420B"
    )

    tmp_script_content_prefix="#!/bin/bash"
    tmp_script_options=(
        "set -o nounset"
        "set -o errexit"
        "set -o pipefail"
    )
    sbatch_options=(
        "-p batch"
        "-t 3-00:00:00"
        "-N 1"
        "--ntasks-per-node=1"
        "--gres=gpu:NVIDIAA100-PCIE-40GB:1"
        "-c 6"
    )

    model_args=(
        "--suite $suite"
        "--force"
        "--sample-size $sample_size"
        "--n-exemplar 0"
    )

    senario="few-shot-syntactic-knowledge"

    tmp_dir=$(mktemp -d)
    echo "tmp_dir: $tmp_dir"

    # write sbatch script
    tmp_script_path="$tmp_dir/revision.sh"
    echo "$tmp_script_content_prefix" > "$tmp_script_path"
    for option in "${sbatch_options[@]}"; do
        echo "#SBATCH $option" >> "$tmp_script_path"
    done
    echo "" >> "$tmp_script_path"
    for option in "${tmp_script_options[@]}"; do
        echo "$option" >> "$tmp_script_path"
    done
    echo "" >> "$tmp_script_path"
    echo "{" >> "$tmp_script_path"
    echo "    cd $PWD" >> "$tmp_script_path"
    echo "    export TRANSFORMERS_OFFLINE=1" >> "$tmp_script_path"
    echo "    . scripts/set_environment.sh" >> "$tmp_script_path"
    echo "" >> "$tmp_script_path"
    echo "    for i in {0..2}; do" >> "$tmp_script_path"

    for revision in "${revisions[@]}"; do
        echo "        python -u test.py --model-name $base_model --model-revision $revision ${model_args[@]} --seed \$i" >> "$tmp_script_path"
    done

    echo "    done" >> "$tmp_script_path"
    echo "" >> "$tmp_script_path"
    echo "}" >> "$tmp_script_path"
    # submit sbatch script
    # check how many jobs are in the queue, wc -l minus 1
    n_jobs_in_queue=$(squeue -u "$USER" | wc -l)
    n_jobs_in_queue=$((n_jobs_in_queue - 1))
    while [ "$n_jobs_in_queue" -gt 9 ]; do
        echo "$n_jobs_in_queue jobs in the queue, wait for 1 minute"
        # sleep 1m
        for i in {60..0}; do
            echo -ne "$i\033[0K\r"
            sleep 1
        done
        n_jobs_in_queue=$(squeue -u "$USER" | wc -l)
        n_jobs_in_queue=$((n_jobs_in_queue - 1))
    done
    echo "Submit $tmp_script_path"
    sbatch "$tmp_script_path"


}
