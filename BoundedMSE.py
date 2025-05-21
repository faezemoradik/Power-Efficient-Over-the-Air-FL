import numpy as np
import cvxpy as cp
from GSDSandMMSE import MMSE
import math

#------------------------------------------------------------------------
def OptimizingTransmitScalars(model_size, channel_matrix, num_of_clients, clients_num_sample, z, f_prime, MSE_bound, P_zero, noise_variance):
  stop_flag = False
  K = np.sum(clients_num_sample)
  Lambda= clients_num_sample*z
  vec_phase= np.angle(channel_matrix@np.conj(f_prime))
  vec_abs= np.abs(channel_matrix@np.conj(f_prime))

  b= cp.Variable(num_of_clients, nonneg=True)
  objective_function= cp.norm(b, 2)
  constraints=[]  
  b_upp_bound= np.minimum( np.sqrt(P_zero)*np.ones(num_of_clients), Lambda/vec_abs)
  constraints.append( b <= b_upp_bound)

  rhs = (K**2)*MSE_bound - (model_size)*0.5*noise_variance*(np.linalg.norm(f_prime)**2)
  norm_of_vec= np.linalg.norm(vec_abs)
  normalized_vec= vec_abs/norm_of_vec
  constraints.append( ((np.sqrt(model_size)*np.sum(Lambda))- np.sqrt(rhs))/(norm_of_vec*np.sqrt(model_size)) <=  normalized_vec@b )

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
def OptimizingReceiveBeamforming(channel_matrix, noise_variance, num_of_clients, clients_num_sample, num_of_antennas, z, Transmit_scalars):
  stop_flag = False
  Lambda= clients_num_sample*z
  H = channel_matrix*Transmit_scalars[:, None]

  r= cp.Variable(num_of_clients, nonneg= True)
  f_prime = cp.Variable(num_of_antennas, complex = True)
  constraints=[]
  constraints.append(Lambda-cp.real(H@cp.conj(f_prime)) <= r)
  constraints.append(cp.real(H@cp.conj(f_prime))-Lambda <= r)

  objective_function= 1e-2*cp.quad_form(f_prime, np.eye(num_of_antennas))+(1e-2/(0.5*noise_variance))*cp.quad_form(r, np.ones((num_of_clients, num_of_clients)))

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
def ConstraintsValidation(model_size, channel_matrix, num_of_clients, clients_num_sample, z, Transmit_scalars, f_prime, noise_variance, MSE_bound, P_zero, printstat):
  K= np.sum(clients_num_sample)
  p = np.zeros(num_of_clients)

  for m in range(num_of_clients):
    a= np.real((channel_matrix[m]@np.conj(f_prime))*Transmit_scalars[m])
    p[m]= np.abs((clients_num_sample[m])*z[m] - a)

  constraint_lhs = (model_size/(K**2))*( math.pow(np.sum(p), 2) + 0.5*noise_variance*math.pow(np.linalg.norm(f_prime), 2 ) )
  constraint_res = constraint_lhs - MSE_bound

  if printstat==True:
    print('Constraint LHS: ', constraint_lhs, 'Constraint RHS: ', MSE_bound)
    print('Constraint Res: ', constraint_res)
    print('1st term: ', (model_size/(K**2))*math.pow(np.sum(p), 2) )
    print('2nd term: ', (model_size/(K**2))*0.5*noise_variance*math.pow(np.linalg.norm(f_prime), 2 ))
    print('------------------------------------------------------------------')
    print('max power: ', max(np.abs(Transmit_scalars))**2, 'Power limit: ', P_zero )

  val1= ( math.pow(np.sum(p), 2) )*(model_size/(math.pow(K,2)))
  scaling= np.sqrt( (MSE_bound - val1)*(K**2)/(model_size*0.5*noise_variance))/np.linalg.norm(f_prime)

  return scaling
#------------------------------------------------------------------------
def PowerMinimizationwithBoundedMSE(model_size, channel_matrix, scaled_channel_matrix, noise_variance, num_of_clients, num_of_antennas, clients_num_sample, grad_variance, MSE_bound, P_zero):
  ######## Finding an initial faesible point by MMSE solution ##########
  ####### scaling channel matrix to make the optimization numerically stable #######
  myscale=1.0*(1e5/1e7)
  my_scaled_channel_matrix = myscale*scaled_channel_matrix
  f_prime, Transmit_scalars= MMSE(my_scaled_channel_matrix, channel_matrix, grad_variance, P_zero, num_of_antennas, num_of_clients, clients_num_sample)

  ####### feasibility check #######
  ####### If the MMSE method cannot find a feasible point, we use the solution of MMSE to not exceed the power limit. #######
  K = np.sum(clients_num_sample)
  rhs = (K**2)*MSE_bound - (model_size)*0.5*noise_variance*(np.linalg.norm(f_prime)**2)
  if rhs < 0 :
    print('rhs is negative!', rhs)
    print('Initial beamforming vector is infeasible')
    return f_prime, Transmit_scalars
  ##################################
  candid_f_prime = 1*f_prime

  ### If the MMSE solution is feasible, we start the alternating optimization process. ###
  ### stop_flag is defined when solver is unable to solve each subproblem, but the solution exists. ###
  for i in range(15):
    print('----------------- optimizing trasnsmit scalars ------------------ ')
    candid_Transmit_scalars, stop_flag = OptimizingTransmitScalars(model_size, channel_matrix, num_of_clients, clients_num_sample, np.sqrt(grad_variance), candid_f_prime, MSE_bound, P_zero, noise_variance)
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
    candid_f_prime, stop_flag = OptimizingReceiveBeamforming(channel_matrix, noise_variance, num_of_clients, clients_num_sample, num_of_antennas, np.sqrt(grad_variance), Transmit_scalars)
    print('stop_flag: ', stop_flag)
    if not stop_flag:
      pass
    else:
      break

    old_obj_value = 1*new_obj_value


  print('----------------- final scaling ------------------ ')
  scaling= ConstraintsValidation(model_size, channel_matrix, num_of_clients, clients_num_sample, np.sqrt(grad_variance), Transmit_scalars, f_prime, noise_variance, MSE_bound, P_zero, True)
  print('Scaling is: ', scaling)
  f_prime = f_prime*scaling
  Transmit_scalars = Transmit_scalars/scaling
  print('---- Final Objective value (total power consumption): -----', np.sum(np.abs(Transmit_scalars)**2))


  return f_prime, Transmit_scalars

#--------------------------------------------------------------------------

