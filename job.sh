#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:50:00
#SBATCH --account=def-benliang
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --mail-user=<faeze.moradi@mail.utoronto.ca>
#SBATCH --mail-type=ALL

module load python/3.11.5
source ~/projects/def-benliang/kalarde/powenv/bin/activate
cd ~/projects/def-benliang/kalarde/PowerMinimizationFL


python main.py -learning_rate ${LR} -alpha ${ALPHA} -delta ${DELTA} -beta 0.0 -batch_size ${BS} -num_epoch ${NE} \
-epsilon ${EPS} -myseed ${MS} -dataset 'CIFAR10' -method ${ME} \
-P_zero ${POLI} -MSE_bound ${MSEB}
