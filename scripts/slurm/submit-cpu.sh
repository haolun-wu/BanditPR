#!/bin/bash
#SBATCH --time=5:0:0
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu
#SBATCH --mem=16G
#SBATCH --output=./outputs/output_%j.txt
#SBATCH --error=./outputs/error_%j.txt

while [[ $# -gt 0 ]]; do
    case "$1" in
        --retriever)
            retriever=$2
            shift 2
            ;;
        --tasks)
            tasks=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

source /home/mila/h/haolun.wu/.bashrc
source /home/mila/h/haolun.wu/projects/environment/BanditPR/bin/activate
bash scripts/$retriever.sh --tasks $tasks
