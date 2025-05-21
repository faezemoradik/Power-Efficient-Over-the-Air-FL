#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --account=def-benliang
#SBATCH --mail-user=faeze.moradi@mail.utoronto.ca
#SBATCH --mail-type=ALL



# for lr in 0.2; do
#   for bs in 10; do
#     for al in 0.05; do
#       for del in 0.05; do
#         for ms in 0 1 2 3 4; do
#           for poli in 15 18 21 24 27; do
#             for eps in 0.0; do
#               for ne in 5; do
#                 for mseb in 0.4; do
#                   for me in 'PoMFL'; do
#                     sbatch --export=LR=${lr},BS=${bs},ALPHA=${al},DELTA=${del},MS=${ms},POLI=${poli},EPS=${eps},NE=${ne},MSEB=${mseb},ME=${me} job.sh
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


for lr in 0.2; do
  for bs in 10; do
    for al in 0.05; do
      for del in 0.05; do
        for ms in 0 1 2 3 4; do
          for poli in 15 18 21 24 27; do
            for eps in 0.0; do
              for ne in 5; do
                for mseb in 0.4; do
                  for me in 'BoundedMSE'; do
                    sbatch --export=LR=${lr},BS=${bs},ALPHA=${al},DELTA=${del},MS=${ms},POLI=${poli},EPS=${eps},NE=${ne},MSEB=${mseb},ME=${me} job.sh
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# for lr in 0.01; do
#   for bs in 10; do
#     for al in 0.05; do
#       for del in 0.05; do
#         for ms in 0 1 2 3 4; do
#           for poli in 15 18 21 24 27; do
#             for eps in 0.0; do
#               for ne in 5; do
#                 for mseb in 0.4; do
#                   for me in 'GSDS'; do
#                     sbatch --export=LR=${lr},BS=${bs},ALPHA=${al},DELTA=${del},MS=${ms},POLI=${poli},EPS=${eps},NE=${ne},MSEB=${mseb},ME=${me} job.sh
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


# for lr in 0.01; do
#   for bs in 10; do
#     for al in 0.05; do
#       for del in 0.05; do
#         for ms in 3; do
#           for poli in 15; do
#             for eps in 0.0; do
#               for ne in 5; do
#                 for mseb in 0.4; do
#                   for me in 'MMSE'; do
#                     sbatch --export=LR=${lr},BS=${bs},ALPHA=${al},DELTA=${del},MS=${ms},POLI=${poli},EPS=${eps},NE=${ne},MSEB=${mseb},ME=${me} job.sh
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done

