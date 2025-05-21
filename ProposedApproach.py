import numpy as np
import math
import cvxpy as cp
from GSDSandMMSE import MMSE


#------------------------------------------------------------------------
def OptimizingTransmitScalars(model_size, channel_matrix, noise_variance, num_of_clients, clients_num_sample, z, scaling_factor, f_prime, delta, alpha, beta, global_gradient_norm, P_zero, epsilon, path_losses, CSA):
  stop_flag = False
  K = np.sum(clients_num_sample)
  Lambda= clients_num_sample*z
  vec_abs= np.abs(channel_matrix@np.conj(f_prime))
  vec_phase= np.angle(channel_matrix@np.conj(f_prime))
  global_gradient_norm = np.array(global_gradient_norm)
  ones_matrix = np.ones((num_of_clients, num_of_clients))

  b= cp.Variable(num_of_clients, nonneg=True)
  objective_function= cp.norm(b, 2)/(scaling_factor)
  constraints=[]  

  b_upp_bound= np.minimum( np.sqrt(P_zero)*np.ones(num_of_clients), Lambda/vec_abs)
  constraints.append( b <= b_upp_bound)

  rhs = (K**2)*(delta*(global_gradient_norm**2)+beta) - (model_size)*0.5*noise_variance*(np.linalg.norm(f_prime)**2)

  if CSA == 'perfect':
    R= 1*min( np.sqrt(rhs), K*alpha*global_gradient_norm ) 
    norm_of_vec= np.linalg.norm(vec_abs)
    normalized_vec= vec_abs/norm_of_vec
    constraints.append( ((np.sqrt(model_size)*np.sum(Lambda))-R )/(norm_of_vec*np.sqrt(model_size)) <= normalized_vec@b )

  elif CSA == 'imperfect':
    path_loss_matrix= np.diag(1/path_losses)
    constraints.append(  (np.sqrt(model_size)*np.sum(Lambda))- (K*alpha*global_gradient_norm) <= np.sqrt(model_size)*(vec_abs@b)  )
    constraints.append(model_size*( cp.quad_form(Lambda-cp.multiply(vec_abs, b), ones_matrix)+ 0.5*epsilon*(np.linalg.norm(f_prime)**2)*cp.quad_form(b, path_loss_matrix) ) <= rhs)

  prob = cp.Problem(cp.Minimize(objective_function), constraints)
  try:
    prob.solve(cp.CVXOPT)
    if (prob.status != 'optimal'):
      prob.solve(solver= cp.SCS)
      if (prob.status != 'optimal'):
        print('Problem status is:', prob.status)
        stop_flag = True
        output = None 
      else:
        output = b.value*np.exp(-1j*vec_phase )
    else:
      output = b.value*np.exp(-1j*vec_phase )
  except:
    print('Solver Error')
    stop_flag = True
    output = None 

  return output, stop_flag
#--------------------------------------------------------------------------
def OptimizingReceiveBeamforming(model_size, channel_matrix, noise_variance, num_of_clients, clients_num_sample, num_of_antennas, z, Transmit_scalars, alpha, global_gradient_norm, path_losses, epsilon, CSA):
  stop_flag = False
  K = np.sum(clients_num_sample) 
  Lambda= clients_num_sample*z
  global_gradient_norm = np.array(global_gradient_norm)
  H = channel_matrix*Transmit_scalars[:, None]


  r= cp.Variable(num_of_clients, nonneg= True)
  f_prime = cp.Variable(num_of_antennas, complex = True)
  constraints=[]
  constraints.append(Lambda-cp.real(H@cp.conj(f_prime)) <= r)
  constraints.append(cp.real(H@cp.conj(f_prime))-Lambda <= r)
  up_bound_1= K*alpha*global_gradient_norm
  constraints.append( np.sqrt(model_size)*cp.sum(r) <= up_bound_1 )

  if CSA == 'perfect':
    objective_function= 1e2*cp.quad_form(f_prime, np.eye(num_of_antennas))+(1e2/(0.5*noise_variance))*cp.quad_form(r, np.ones((num_of_clients, num_of_clients)))
  elif CSA == 'imperfect':
    channel_vars= 1/path_losses
    effective_var= noise_variance + epsilon* channel_vars@(np.abs(Transmit_scalars)**2)
    objective_function= 1e-6*cp.quad_form(f_prime, np.eye(num_of_antennas))+(1e-6/(0.5*effective_var))*cp.quad_form(r, np.ones((num_of_clients, num_of_clients)))
    
  prob = cp.Problem(cp.Minimize(objective_function), constraints)

  try:
    prob.solve(cp.CVXOPT)
    if (prob.status != 'optimal'):
      prob.solve(solver= cp.SCS)
      if (prob.status != 'optimal'):
        print('Problem status is:', prob.status)
        stop_flag = True
        output = None 
      else:
        output = f_prime.value
    else:
      output = f_prime.value
  except:
    print('Solver Error')
    stop_flag = True
    output = None 



  return output, stop_flag

#----------------------------------------------------------------------
def ConstraintsValidation(P_zero, model_size, channel_matrix, noise_variance , num_of_clients, clients_num_sample, z, Transmit_scalars, f_prime, global_gradient_norm, delta, beta, alpha, epsilon, path_losses, CSA, printstat):
  K= np.sum(clients_num_sample)
  p = np.zeros(num_of_clients)
  global_gradient_norm = np.array(global_gradient_norm)

  for m in range(num_of_clients):
    a= np.real((channel_matrix[m]@np.conj(f_prime))*Transmit_scalars[m])
    p[m]= np.abs((clients_num_sample[m])*z[m] - a)

  constraint_1_res = (np.sqrt(model_size)/K)*np.sum(p)- (alpha*global_gradient_norm)

  if CSA == 'perfect':
    constraint_2_lhs = (model_size/(K**2))*( math.pow(np.sum(p), 2) + 0.5*noise_variance*math.pow(np.linalg.norm(f_prime), 2 ) )


  elif CSA == 'imperfect':
    path_loss_matrix= np.diag(1/path_losses)
    transmit_scalars_abs= np.abs(Transmit_scalars)
    third_term= transmit_scalars_abs@path_loss_matrix@transmit_scalars_abs
    constraint_2_lhs = (model_size/(K**2))*( math.pow(np.sum(p), 2) + 0.5*noise_variance*math.pow(np.linalg.norm(f_prime), 2 )+ 0.5*epsilon*math.pow(np.linalg.norm(f_prime), 2 )*third_term )

  constraint_2_rhs= delta*(global_gradient_norm**2)+beta
  constraint_2_res = constraint_2_lhs - constraint_2_rhs

  if printstat==True:
    print('Constraint 1 LHS: ', (np.sqrt(model_size)/K)*np.sum(p), 'Constraint 1 RHS:', (alpha*global_gradient_norm))
    print('Constraint 1 Res: ', constraint_1_res)
    print('------------------------------------------------------------------')
    print('Constraint 2 LHS: ', constraint_2_lhs, 'Constraint 2 RHS: ', constraint_2_rhs)
    print('Constraint 2 Res: ', constraint_2_res)
    print('1st term: ', (model_size/(K**2))*math.pow(np.sum(p), 2) )
    print('2nd term: ', (model_size/(K**2))*0.5*noise_variance*math.pow(np.linalg.norm(f_prime), 2 ))
    if CSA == 'imperfect':
      print('3rd term: ', (model_size/(K**2))*0.5*epsilon*math.pow(np.linalg.norm(f_prime), 2 )*third_term)
    print('------------------------------------------------------------------')
    print('max power: ', max(np.abs(Transmit_scalars))**2, 'Power limit: ', P_zero )


  if CSA == 'perfect':
    val1= ( math.pow(np.sum(p), 2) )*(model_size/(math.pow(K,2)))

  elif CSA == 'imperfect':
    val1= (math.pow(np.sum(p), 2)+ 0.5*epsilon*math.pow(np.linalg.norm(f_prime), 2 )*third_term )*(model_size/(math.pow(K,2)))
  
  
  scaling= np.sqrt( (constraint_2_rhs - val1)*(K**2)/(model_size*0.5*noise_variance))/np.linalg.norm(f_prime)


  return scaling

#----------------------------------------------------------------------------------
def OurProposedMethod(model_size, channel_matrix, scaled_channel_matrix, noise_variance, num_of_clients, num_of_antennas, clients_num_sample, grad_variance, delta, alpha, beta, global_gradient_norm, P_zero, epsilon, path_losses, CSA):

  ######## Finding an initial faesible point by MMSE solution ##########
  ####### scaling channel matrix to make the optimization numerically stable #######
  myscale=1.0*(1e5/1e7)
  my_scaled_channel_matrix = myscale*scaled_channel_matrix
  f_prime, Transmit_scalars= MMSE(my_scaled_channel_matrix, channel_matrix, grad_variance, P_zero, num_of_antennas, num_of_clients, clients_num_sample)

  ####### feasibility check #######
  ####### If the MMSE method cannot find a feasible point, we use the solution of MMSE to not exceed the power limit. #######
  K = np.sum(clients_num_sample)
  rhs = (K**2)*(delta*(global_gradient_norm**2)+beta) - (model_size)*0.5*noise_variance*(np.linalg.norm(f_prime)**2)
  if rhs < 0 :
    print('RHS is negative!', rhs)
    print('Initial point is infeasible.')
    return f_prime, Transmit_scalars
  #################################
  candid_f_prime = 1*f_prime
    
  ### If the MMSE solution is feasible, we start the alternating optimization process. ###
  ### stop_flag is defined when solver is unable to solve, but the solution exists. ###
  for i in range(15):
    print('----------------- optimizing transmit scalars ------------------ ')
    candid_Transmit_scalars, stop_flag = OptimizingTransmitScalars(model_size, channel_matrix, noise_variance, num_of_clients, clients_num_sample, np.sqrt(grad_variance), 1e0, candid_f_prime, delta, alpha, beta, global_gradient_norm, P_zero, epsilon, path_losses, CSA)
    print('stop_flag: ', stop_flag)
    if not stop_flag:
      Transmit_scalars = 1*candid_Transmit_scalars
      f_prime = 1*candid_f_prime
    else:
      break

    new_obj_value = np.sum(np.abs(Transmit_scalars)**2)
    print('---- Objective value (power consumption): -----', new_obj_value)
    ### stopping criteria
    if i > 0:
      if abs(old_obj_value - new_obj_value )/old_obj_value < 0.01:
        print('stop after ', i+1, ' iterations')
        break

    print('----------------- optimizing beamforming ------------------ ')
    candid_f_prime, stop_flag = OptimizingReceiveBeamforming(model_size, channel_matrix, noise_variance, num_of_clients, clients_num_sample, num_of_antennas, np.sqrt(grad_variance), Transmit_scalars, alpha, global_gradient_norm, path_losses, epsilon, CSA)
    print('stop_flag: ', stop_flag)
    if not stop_flag:
      pass
    else:
      break

    old_obj_value = 1*new_obj_value

  ### This setp is the final scaling step in PoMFL algorithm. ###
  print('----------------- final scaling ------------------ ')
  scaling= ConstraintsValidation(P_zero, model_size, channel_matrix, noise_variance, num_of_clients, clients_num_sample, np.sqrt(grad_variance), Transmit_scalars, f_prime, global_gradient_norm, delta, beta, alpha, epsilon, path_losses, CSA,  True)
  print('Scaling is: ', scaling)
  f_prime = f_prime*scaling
  Transmit_scalars = Transmit_scalars/scaling
  print('---- Final Objective value (total power consumption): -----', np.sum(np.abs(Transmit_scalars)**2))

  return f_prime, Transmit_scalars
#----------------------------------------------------------------------------------
