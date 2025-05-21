
from mymain import main
import argparse
import numpy as np
import pickle


for seed in [0]:
    args = argparse.Namespace(learning_rate= 0.95, P_zero = 15, alpha=0.55, delta=6.0, 
                                            beta = 0.0, MSE_bound= 0.085, batch_size=5420, 
                                            num_epoch= 20, myseed= seed, epsilon=0.0, 
                                            dataset='MNIST', method='PoMFL')

    main(args)   