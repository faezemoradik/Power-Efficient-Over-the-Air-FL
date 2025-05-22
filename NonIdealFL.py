
import torch
from torch import nn
import numpy as np
import sys
from models import ResNet, LogisticRegression, BasicBlock
from GSDSandMMSE import GSDS, ReceiveTransmitBeamforming, MMSE
# from BoundedBias import PowerMinimizationwithBoundedBias
from BoundedMSE import PowerMinimizationwithBoundedMSE
from ProposedApproach import OurProposedMethod
from channel import Channel_Condition, clients_distance


##################################################
def validation(model, input, target, criterion, mode):
  model.train(mode= mode)
  with torch.no_grad():
    output = model(input)
    loss = criterion(output, target)
    prediction = output.argmax(dim=1, keepdim=True)
    correct = prediction.eq(target.view_as(prediction)).sum().item()

  acc= correct/ len(input)

  return loss.item() , acc
##################################################
def train_clients(model, inputs, targets, criterion, optimizer):
  model.train()
  output = model(inputs)
  loss = criterion(output, targets)
  optimizer.zero_grad()
  loss.backward()
  grad_vector=[]
  for param in model.parameters():
    grad_vector.append(param.grad) 

  return grad_vector
##########################################
def create_grads_dict(num_of_clients , model_dict, criterion_dict , optimizer_dict, x_train_dict, y_train_dict, i,  batch_size):
  grads_dict = dict()
  for j in range(num_of_clients):
    model= model_dict['model'+ str(j)]
    criterion= criterion_dict['criterion'+ str(j)]
    optimizer= optimizer_dict['optimizer'+ str(j)]

    ind = i%(int(len(x_train_dict['client'+str(j)])/batch_size))

    xt= x_train_dict['client'+str(j)][ind*batch_size:(ind+1)*batch_size]
    yt= y_train_dict['client'+str(j)][ind*batch_size:(ind+1)*batch_size]

    grad= train_clients(model, xt,  yt, criterion, optimizer)

    grads_dict.update({'client'+str(j) : grad })

  return grads_dict
#########################################
def l2_norm_calculator(u):
  return np.linalg.norm(torch.nn.utils.parameters_to_vector(u).cpu().numpy())

########################################
def create_model_optimizer_criterion_dict(num_of_clients, learning_rate, model_name):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  model_dict = dict()
  optimizer_dict= dict()
  criterion_dict = dict()
    
  for i in range(num_of_clients):

    if model_name == 'LogisticRegression':
        model_info = LogisticRegression().to(device)
        optimizer_info= torch.optim.SGD(model_info.parameters(), lr=learning_rate, weight_decay=0.0001)
    elif model_name == 'ResNet':
        model_info = ResNet(BasicBlock, [2,2,2]).to(device)
        optimizer_info= torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        
    criterion_info = nn.CrossEntropyLoss().to(device)

    model_dict.update({"model"+str(i) : model_info })
    optimizer_dict.update({"optimizer"+str(i) : optimizer_info })
    criterion_dict.update({"criterion"+str(i) : criterion_info})
        
  return model_dict, optimizer_dict, criterion_dict 

########################################################################
def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, num_of_clients):
  state_dict = main_model.state_dict()
  filtered_state_dict = {k: v for k, v in state_dict.items() if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}
  for i in range(num_of_clients):
    model_dict['model'+str(i)].load_state_dict(filtered_state_dict, strict=False)
    # model_dict['model'+str(i)].load_state_dict(state_dict)

  return model_dict
############################################################################
def Global_loss_gradient(grads_dict, num_of_clients, clients_num_sample):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  grads_mean= []
  K = np.sum(clients_num_sample)

  with torch.no_grad():   
    for j in range(len(grads_dict['client0'])):
      grads_mean.append(torch.zeros(size= grads_dict['client0'][j].size()).to(device))
      for i in range(num_of_clients):
        grads_mean[j] = grads_mean[j]+ clients_num_sample[i]*grads_dict['client'+str(i)][j]

      grads_mean[j] = grads_mean[j]/K

  return grads_mean
##########################################################################
def Update_main_model(main_model, update_params, main_optimizer):
  main_model.train()
  i=0
  for p in main_model.parameters():
    p.grad = update_params[i]
    i +=1

  main_optimizer.step()
  return main_model
#######################################################
def GradNormSquared(grads_dict, num_of_clients):
  model_size = 0
  for j in range(len(grads_dict['client0'])):
    model_size = model_size + grads_dict['client0'][j].numel()

  grads_norm_squared= np.zeros(num_of_clients)
  for i in range(num_of_clients):
    for j in range(len(grads_dict['client0'])):
      grads_norm_squared[i] = grads_norm_squared[i]+ torch.sum((grads_dict['client'+str(i)][j])**2)

    grads_norm_squared[i]= grads_norm_squared[i]/model_size

  return grads_norm_squared, model_size

##########################################################
def Update_Calculator(grads_dict, num_of_clients, clients_num_sample, noise , Transmit_scalars, f_prime, H ):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  grads_mean= []
  client_selection_array= np.ones(num_of_clients)

  for j in range(num_of_clients):
    if np.abs(Transmit_scalars[j]) == 0:
      client_selection_array[j] = 0
    
  K = clients_num_sample@client_selection_array
  grad_variances, _ = GradNormSquared(grads_dict, num_of_clients)

  ##### effective_noise: receiver noise after post processing at server
  noise_term= np.real( noise @ np.conj(f_prime) ) /(K)
  noise_term= noise_term.astype('float32')
  effective_noise=[]
  start_point=0
  for j in range(len(grads_dict['client0'])):
    interval= grads_dict['client0'][j].numel()
    effective_noise.append(np.reshape(noise_term[start_point: start_point+ interval ], grads_dict['client0'][j].size() ))
    start_point += interval

  ##### grads_mean: the post processed signal at the server used for updating the main model 
  with torch.no_grad():
    for j in range(len(grads_dict['client0'])):
      grads_mean.append( torch.zeros(size= grads_dict['client0'][j].size()).to(device) )
      for i in range(num_of_clients):
        grads_mean[j] = grads_mean[j]+ np.real(Transmit_scalars[i]*(np.conj(f_prime)@H[i]))* grads_dict['client'+str(i)][j]/np.sqrt(grad_variances[i])

      effective_noise[j]= torch.tensor(effective_noise[j]).to(device)
      grads_mean[j] = grads_mean[j]/K + effective_noise[j]

  return grads_mean
#############################################################
def LoaderValidation(model, criterion, loader, mode):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  loader_loss=0
  loader_acc=0
  for j, (batch_data, batch_labels) in enumerate(loader):
    batch_data= batch_data.to(device)
    batch_labels = batch_labels.to(device)
    loss, acc= validation(model, batch_data, batch_labels, criterion, mode)
    loader_loss += loss*batch_data.size(0)
    loader_acc += acc*batch_data.size(0)

  loader_loss /= len(loader.dataset)
  loader_acc /= len(loader.dataset)

  return loader_loss, loader_acc

##############################################################################
def NonIdealFedSGD(x_train_dict, y_train_dict, train_loader, test_loader, model_name, num_epoch, num_of_clients, clients_num_sample, num_of_antennas, 
                     learning_rate, r_min, r_max, P_zero, noise_variance, batch_size, method, alpha, delta, beta, MSE_bound, epsilon, target_acc):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  if model_name == 'LogisticRegression':
    main_model = LogisticRegression().to(device)
    main_optimizer= torch.optim.SGD(main_model.parameters(), lr=learning_rate, weight_decay=0.0001)
    flag = False
  elif model_name == 'ResNet':
    main_model = ResNet(BasicBlock, [2,2,2]).to(device)
    main_optimizer= torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    flag = True

  main_criterion = nn.CrossEntropyLoss().to(device)
  model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(num_of_clients, learning_rate, model_name)

  train_acc_list = np.zeros(num_epoch+1)
  test_acc_list = np.zeros(num_epoch+1)
  train_loss_list = np.zeros(num_epoch+1)
  test_loss_list = np.zeros(num_epoch+1)
  num_of_comm_round_in_each_epoch = int(clients_num_sample[0]/batch_size)
  transmit_power_list = np.zeros((num_epoch, num_of_comm_round_in_each_epoch,  num_of_clients))

  train_loss, train_acc= LoaderValidation(main_model, main_criterion, train_loader, flag)
  test_loss, test_acc= LoaderValidation(main_model, main_criterion, test_loader, flag)
  train_loss_list[0] = train_loss
  train_acc_list[0] = train_acc
  test_loss_list[0] = test_loss
  test_acc_list[0] = test_acc

  print("Epoch", str(0), "| Test loss: ", test_loss, "Test acc :", test_acc, "Train loss: " , train_loss, "Train acc :", train_acc )
  clients_distances = clients_distance(r_min, r_max, num_of_clients)

  for i in range(num_epoch):
    for j in range(num_of_comm_round_in_each_epoch):
      print('Communication round', str(j+1), ':')

      ### random channel generation
      channel_estimation_matrix, true_channel_matrix, scaled_channel_estimation_matrix, path_losses = Channel_Condition(clients_distances, num_of_clients, num_of_antennas, clients_num_sample, epsilon)

      model_dict= send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, num_of_clients)
      grads_dict= create_grads_dict(num_of_clients, model_dict, criterion_dict, optimizer_dict, x_train_dict, y_train_dict, j, batch_size)

      grad_variance, model_size = GradNormSquared(grads_dict, num_of_clients)

      if method== 'GSDS':
        f_beam, devices_selection_array = GSDS(scaled_channel_estimation_matrix, channel_estimation_matrix, noise_variance, P_zero, num_of_antennas, num_of_clients, clients_num_sample)
        f_prime, Transmit_scalars = ReceiveTransmitBeamforming(grad_variance, channel_estimation_matrix, P_zero, num_of_clients, clients_num_sample, f_beam, devices_selection_array)

      elif method == 'MMSE': 
        f_prime, Transmit_scalars = MMSE(scaled_channel_estimation_matrix, channel_estimation_matrix, grad_variance, P_zero, num_of_antennas, num_of_clients, clients_num_sample)

      elif method == 'PoMFL':
        CSA = 'perfect'
        global_gradient_norm_estimate= np.sqrt(model_size)*(clients_num_sample@np.sqrt(grad_variance))/(np.sum(clients_num_sample))
        f_prime, Transmit_scalars = OurProposedMethod(model_size, channel_estimation_matrix, scaled_channel_estimation_matrix, 
                                                      noise_variance, num_of_clients, num_of_antennas, clients_num_sample, 
                                                      grad_variance, delta, alpha, beta, global_gradient_norm_estimate, P_zero, None, None, CSA)
        
      elif method == 'PoMFL-ImCSI':
        CSA = 'imperfect'
        global_gradient_norm_estimate= np.sqrt(model_size)*(clients_num_sample@np.sqrt(grad_variance))/(np.sum(clients_num_sample))
        f_prime, Transmit_scalars = OurProposedMethod(model_size, channel_estimation_matrix, scaled_channel_estimation_matrix, 
                                                      noise_variance, num_of_clients, num_of_antennas, clients_num_sample, 
                                                      grad_variance, delta, alpha, beta, global_gradient_norm_estimate, P_zero, epsilon, path_losses, CSA)
        
      elif method == 'BoundedMSE':
        f_prime, Transmit_scalars = PowerMinimizationwithBoundedMSE(model_size, channel_estimation_matrix, scaled_channel_estimation_matrix, 
                                                                    noise_variance, num_of_clients, num_of_antennas, clients_num_sample, 
                                                                    grad_variance, MSE_bound, P_zero)
        

      transmit_power_list[i,j] = np.abs(Transmit_scalars)**2

      noise = np.random.normal(0, np.sqrt(noise_variance)/np.sqrt(2), (model_size, num_of_antennas))+1j*np.random.normal(0, np.sqrt(noise_variance)/np.sqrt(2), (model_size, num_of_antennas))

      update = Update_Calculator(grads_dict, num_of_clients, clients_num_sample, noise, Transmit_scalars, f_prime, true_channel_matrix)
      main_model = Update_main_model(main_model, update, main_optimizer)


    train_loss, train_acc= LoaderValidation(main_model, main_criterion, train_loader, flag)
    test_loss, test_acc= LoaderValidation(main_model, main_criterion, test_loader, flag)
    train_loss_list[i+1] = train_loss
    train_acc_list[i+1] = train_acc
    test_loss_list[i+1] = test_loss
    test_acc_list[i+1] = test_acc

    print("Epoch", str(i+1), "| Test loss: ", test_loss, "Test acc :", test_acc, "Train loss: " , train_loss, "Train acc :", train_acc )

    num_communication_rounds_till_target_acc = (i+1)*num_of_comm_round_in_each_epoch
    if test_acc >= target_acc:
      break

  return train_acc_list , test_acc_list , train_loss_list , test_loss_list, transmit_power_list, num_communication_rounds_till_target_acc, i+1

##############################################################################

