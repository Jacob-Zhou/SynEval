#!/bin/bash
# Path: run_all.sh
# This file generates temperary scripts and submits them to the cluster.

set -o nounset
set -o errexit
set -o pipefail

{

    # key variables
    suite="v23"
    sample_size="5"

    # models that fit in single A100
    single_a100_models=(
        "mistralai/Mistral-7B-v0.1"
        "mistralai/Mistral-7B-Instruct-v0.1"
        "baichuan-inc/Baichuan2-7B-Base"
        "baichuan-inc/Baichuan2-7B-Chat"
        "THUDM/chatglm2-6b"
        "meta-llama/Llama-2-7b-hf"
        "meta-llama/Llama-2-7b-chat-hf"
        "huggyllama/llama-7b"
        "tiiuae/falcon-rw-1b"
        "tiiuae/falcon-7b"
        "tiiuae/falcon-7b-instruct"
    )

    # models that fit in two A100s
    dual_a100_models=(
        "baichuan-inc/Baichuan2-13B-Base"
        "baichuan-inc/Baichuan2-13B-Chat"
        "meta-llama/Llama-2-13b-hf"
        "meta-llama/Llama-2-13b-chat-hf"
        "huggyllama/llama-13b"
        "huggyllama/llama-30b"
    )

    # models that require four A100s
    quad_a100_models=(
        "meta-llama/Llama-2-70b-hf"
        "meta-llama/Llama-2-70b-chat-hf"
        "huggyllama/llama-65b"
        "tiiuae/falcon-40b"
        "tiiuae/falcon-40b-instruct"
    )

    tmp_script_content_prefix="#!/bin/bash"
    tmp_script_options=(
        "set -o nounset"
        "set -o errexit"
        "set -o pipefail"
    )
    sbatch_global_options=(
        "-p batch"
        "-t 3-00:00:00"
        "-N 1"
        "--ntasks-per-node=1"
    )
    sbatch_single_a100_options=(
        "--gres=gpu:NVIDIAA100-PCIE-40GB:1"
        "-c 6"
    )
    sbatch_dual_a100_options=(
        "--gres=gpu:NVIDIAA100-PCIE-40GB:2"
        "-c 12"
    )
    sbatch_quad_a100_options=(
        "--gres=gpu:NVIDIAA100-PCIE-40GB:4"
        "-c 24"
    )

    global_model_args=(
        "--suite $suite"
        "--force"
        "--sample-size $sample_size"
    )

    senarios=(
        "zero-shot"
        "few-shot-syntactic-knowledge"
    )
    zero_shot_args=(
        "--n-exemplar 0"
    )
    few_shot_args=(
        "--n-exemplar 5"
    )
    few_shot_syntactic_knowledge_args=(
        "--n-exemplar 5"
        "--exemplar-type syntactic-knowledge"
    )
    few_shot_exclude_self_syntactic_knowledge_args=(
        "--n-exemplar 5"
        "--exemplar-type exclude-self-syntactic-knowledge"
    )

    tmp_dir=$(mktemp -d)
    echo "tmp_dir: $tmp_dir"

    # for requirement in "single" "dual"; do
    for requirement in "single" "dual" "quad"; do
        if [ "$requirement" == "single" ]; then
            models=("${single_a100_models[@]}")
            sbatch_options=("${sbatch_global_options[@]}" "${sbatch_single_a100_options[@]}")
        elif [ "$requirement" == "dual" ]; then
            models=("${dual_a100_models[@]}")
            sbatch_options=("${sbatch_global_options[@]}" "${sbatch_dual_a100_options[@]}")
        elif [ "$requirement" == "quad" ]; then
            models=("${quad_a100_models[@]}")
            sbatch_options=("${sbatch_global_options[@]}" "${sbatch_quad_a100_options[@]}")
        else
            echo "Unknown requirement: $requirement"
            exit 1
        fi
        for senario in "${senarios[@]}"; do
            # write sbatch script
            tmp_script_path="$tmp_dir/$requirement-$senario.sh"
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
            if [ "$senario" == "zero-shot" ]; then
                echo "    for i in {0..0}; do" >> "$tmp_script_path"
                model_args=("${global_model_args[@]}" "${zero_shot_args[@]}")
            elif [ "$senario" == "few-shot" ]; then
                echo "    for i in {0..2}; do" >> "$tmp_script_path"
                model_args=("${global_model_args[@]}" "${few_shot_args[@]}")
            elif [ "$senario" == "few-shot-syntactic-knowledge" ]; then
                echo "    for i in {0..2}; do" >> "$tmp_script_path"
                model_args=("${global_model_args[@]}" "${few_shot_syntactic_knowledge_args[@]}")
            elif [ "$senario" == "few-shot-exclude-self-syntactic-knowledge" ]; then
                echo "    for i in {0..2}; do" >> "$tmp_script_path"
                model_args=("${global_model_args[@]}" "${few_shot_exclude_self_syntactic_knowledge_args[@]}")
            else
                echo "Unknown senario: $senario"
                exit 1
            fi
            for model in "${models[@]}"; do
                echo "        python -u test.py --model-name $model ${model_args[@]} --seed \$i" >> "$tmp_script_path"
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
        done
    done
}
