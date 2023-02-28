#!/bin/bash
#SBATCH --job-name=DetCo_unsup
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --mem=40G

source /user/home/qc18229/initConda.sh
source activate openmmlab

module load lang/cuda/11.0

cd /user/work/qc18229/DetCo_pytorch

python  main_detco.py \
	-a resnet18 \
	--lr 0.005\
	--batch-size 2\
	--epochs 2 \
        --dist-file '/user/work/qc18229/BPtest' \
	--multiprocessing-distributed \
	--world-size 1 --rank 0 \
	/user/work/qc18229/DetCo.pytorch/LUSDATA \
	--detco-t 0.2 --aug-plus --cos


conda deactivate