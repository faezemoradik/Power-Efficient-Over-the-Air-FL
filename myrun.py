
from mymain import main
import argparse
import numpy as np
import pickle

# MSE_bound for mnist changed to 0.085

for seed in [0]:
    args = argparse.Namespace(learning_rate= 0.2, P_zero = 15, alpha=0.05, delta=0.05, 
                                            beta = 0.0, MSE_bound= 0.4, batch_size=10, 
                                            num_epoch= 5, myseed= seed, epsilon=0.0, 
                                            dataset='CIFAR10', method='BoundedMSE')

    main(args)   