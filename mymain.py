import numpy as np
import torch
import pickle
import os
import argparse
from channel import Channel_Condition
from data import MNIST_PROCESS, CIFAR10_PROCESS
from NonIdealFL import NonIdealFedSGD
import random
import matplotlib.pyplot as plt


#----------------------------------------------------
def get_parameter():
  parser=argparse.ArgumentParser()
  parser.add_argument("-learning_rate",default=0.05,type=float,help="learning rate")
  parser.add_argument("-P_zero",default= 0.5,type=float,help="P_zero")
  parser.add_argument("-batch_size",default= 5,type=int,help="batch size")
  parser.add_argument("-myseed",default=0,type=int,help="seed")
  parser.add_argument("-num_epoch",default=0,type=int,help="number of epoch")
  parser.add_argument("-epsilon",default=0.1,type=float,help="channle error multiplier")
  parser.add_argument("-dataset",default='MNIST',type=str,help="dataset name")
  parser.add_argument("-method",default='GSDS',type=str,help="method name")
  parser.add_argument("-delta",default=6.0,type=float,help="a constant in the PoMFL method")
  parser.add_argument("-alpha",default=0.5,type=float,help="a constant in PoMF method ")
  parser.add_argument("-beta",default=0.5,type=float,help="a constant in the PoMF method")
  parser.add_argument("-MSE_bound",default=0.5,type=float,help="MSE Bound for Bounded MSE method")
  args= parser.parse_args()
  return args

#------------------------------------------------------------
def main(args):
  if torch.cuda.is_available():
    print("CUDA is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  else:
    print("CUDA is not available")

  
  # args = get_parameter()
  learning_rate = args.learning_rate
  P_zero = args.P_zero
  batch_size = args.batch_size
  num_epoch = args.num_epoch
  epsilon= args.epsilon
  alpha = args.alpha
  delta = args.delta
  beta = args.beta
  MSE_bound= args.MSE_bound
  dataset= args.dataset
  method= args.method
  myseed = args.myseed

  P_zero_watt = 1e-3*np.power(10, P_zero/10)


  #--------- Parameters ----------
  num_of_antennas= 16 ### number of antennas at parameter server
  num_of_clients = int(10)### number of devices
  bandwidth= 100e3
  noise_variance_db = 10-174+40+ 10*np.log10(bandwidth)-30
  print('noise var: ', noise_variance_db)
  noise_variance= np.power(10, noise_variance_db/10)
  r_max =100
  r_min =10

  print('power limit: ', P_zero_watt)
  print('learning_rate: ', learning_rate)
  print('batch_size: ', batch_size)
  print('epsilon: ', epsilon)
  print('dataset: ', dataset)
  print('method: ', method)
  print('myseed: ', myseed)
  print('num_epoch: ', num_epoch) 
  print('alpha: ', alpha)
  print('delta: ', delta)
  print('beta: ', beta)
  print('MSE bound: ', MSE_bound)

  torch.backends.cudnn.deterministic=True
  torch.cuda.manual_seed(0) # to get the reproducible results
  random.seed(0) # to get the reproducible results
  torch.manual_seed(0) # to get the reproducible results

  if dataset== 'MNIST':
    x_train_dict, y_train_dict , train_loader, test_loader, clients_num_sample = MNIST_PROCESS(num_of_clients)
    model_name = 'LogisticRegression'
    target_acc = 0.845
  elif dataset== 'CIFAR10':
    x_train_dict, y_train_dict , train_loader, test_loader, clients_num_sample = CIFAR10_PROCESS(num_of_clients)
    model_name = 'ResNet'
    target_acc = 0.695


  performance = dict()

  np.random.seed(myseed)
  torch.random.manual_seed(myseed)

  # channel_estimation_matrix, true_channel_matrix, scaled_channel_estimation_matrix, path_losses = Channel_Condition(r_min, r_max, num_of_clients, num_of_antennas, clients_num_sample, epsilon)
  # Channel_dict[str(myseed)] = true_channel_matrix
  train_acc_list, test_acc_list, train_loss_list, test_loss_list, transmit_powers, num_communication_rounds_till_target_acc, stop_epoch_num = NonIdealFedSGD(x_train_dict, y_train_dict, train_loader, test_loader, model_name, num_epoch,  
                                                                                                                                                                                  num_of_clients, clients_num_sample, num_of_antennas, learning_rate, r_min, r_max, 
                                                                                                                                                                                  P_zero_watt, noise_variance, batch_size, method, alpha, delta, beta,
                                                                                                                                                                                  MSE_bound, epsilon, target_acc)

                                                                                                        
  performance[str(myseed)]= dict()
  performance[str(myseed)]['train_acc'] = train_acc_list
  performance[str(myseed)]['test_acc'] = test_acc_list
  performance[str(myseed)]['train_loss'] = train_loss_list
  performance[str(myseed)]['test_loss'] = test_loss_list
  performance[str(myseed)]['transmit_power'] = transmit_powers
    
  print('--------------------------------------------------')
  print('Sumpower over all rounds till reaching target accuracy: ', np.sum(transmit_powers))
  print('Avg of sumpower over rounds: ', np.sum(transmit_powers)/num_communication_rounds_till_target_acc)
  print('Final test acc: ', test_acc_list[stop_epoch_num])
  print('Final train acc: ', train_acc_list[stop_epoch_num])
  print('--------------------------------------------------')

  with open(method+dataset+'PowerLimit'+str(P_zero)+'seed'+str(myseed)+'LR'+str(learning_rate)+'BS'+str(batch_size)+'NE'+str(num_epoch)+'epsilon'+str(epsilon)+'alpha'+str(alpha)+'delta'+str(delta)+'mse_b'+str(MSE_bound), 'wb') as f:
    pickle.dump(performance, f)

  # plt.figure(figsize=(10, 6))
  # plt.grid()
  # plt.xticks(fontsize=12)
  # plt.yticks(fontsize=12)
  # plt.yscale('log')
  # for i in range(10):
  #   plt.plot(store_seq_obj_vals[i], label='Communication round'+str(i))
  #   plt.xlabel('Iteration of AO', fontsize=12)
  #   plt.ylabel('Total Power of Devices (watt)', fontsize=12)
  #   plt.legend(fontsize=12)
  # plt.savefig('AOConvergencePlot.pdf')
  # plt.show()
    

#------------------------------------------------------------
# main()

