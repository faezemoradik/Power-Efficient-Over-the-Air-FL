import numpy as np
import random
import math

#--------------------------------------------------------------
def clients_distance(r_min , r_max, num_of_clients):
  r = np.random.uniform(low= r_min, high= r_max, size= num_of_clients)  # radius

  return r #clients disances from the PS

#-------------------------------------------------------------
def Channel_Error(num_of_clients, num_of_antennas, path_loss, epsilon):
  channel_error = np.zeros((num_of_clients, num_of_antennas))+ 1j*np.zeros((num_of_clients, num_of_antennas))

  for i in range(num_of_clients):
    channel_error_variance = (1/path_loss[i])*(epsilon)
    channel_err = np.sqrt(channel_error_variance)*(1*np.random.normal(0, 1/np.sqrt(2), num_of_antennas)+1j*np.random.normal(0, 1/np.sqrt(2), num_of_antennas))
    channel_error[i] = channel_err

  return channel_error
#-------------------------------------------------------------
def Channel_Estimation(num_of_clients, num_of_antennas, path_loss, epsilon):
  channel_estimation = np.zeros((num_of_clients, num_of_antennas))+ 1j*np.zeros((num_of_clients, num_of_antennas))

  for i in range(num_of_clients):
    channel_estimation_variance = (1/path_loss[i])*(1-epsilon)
    channel_estim = np.sqrt(channel_estimation_variance)*(1*np.random.normal(0, 1/np.sqrt(2), num_of_antennas)+1j*np.random.normal(0, 1/np.sqrt(2), num_of_antennas))
    channel_estimation[i] = channel_estim

  return channel_estimation
#-------------------------------------------------------------
def Channel_Condition(clients_distances, num_of_clients, num_of_antennas, clients_num_sample, epsilon):
  scaled_channel_estimation_matrix= np.zeros((num_of_clients, num_of_antennas))+ 1j*np.zeros((num_of_clients, num_of_antennas))

  c1= 3.2*((np.log10(11.75*1))**2)-4.97
  c2=(44.9-6.55*np.log10(30))*np.log10(clients_distances/1e3)
  path_loss_db= 46.3+33.9*np.log10(2000)-13.82*np.log10(30)-c1+c2
  path_loss= np.power(10, path_loss_db/10)

  channel_estimation_matrix= Channel_Estimation(num_of_clients, num_of_antennas, path_loss, epsilon)
  channel_error_matrix= Channel_Error(num_of_clients, num_of_antennas, path_loss, epsilon)
  true_channel_matrix= channel_estimation_matrix- channel_error_matrix

  for i in range(num_of_clients):
    scaled_channel_estimation_matrix[i] = 1e7*channel_estimation_matrix[i]/clients_num_sample[i]

  scaled_channel_estimation_matrix= scaled_channel_estimation_matrix.T
  
  return channel_estimation_matrix, true_channel_matrix, scaled_channel_estimation_matrix, path_loss
#-------------------------------------------------------------
