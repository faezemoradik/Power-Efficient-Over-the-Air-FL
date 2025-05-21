import numpy as np
import math
import cvxpy as cp


#---------------------------------------------------------------------------
def ReceiveTransmitBeamforming(grad_variance, channel_matrix, P_zero, num_of_clients, clients_num_sample, f_beam, devices_selection_array):
  eta_list=[]
  transmitt_scalar=[]
  for i in range(num_of_clients):
    if devices_selection_array[i]==1:
      fh= np.conjugate(f_beam)@channel_matrix[i]
      fh_norm= np.abs(fh)
      eta_list.append( P_zero*(fh_norm**2)/((clients_num_sample[i]**2)*grad_variance[i]) )
  eta= min(eta_list)

  for b in range(num_of_clients):
    if devices_selection_array[b]==1:
      fh= np.conjugate(f_beam)@channel_matrix[b]
      transmitt_scalar.append( clients_num_sample[b]* np.sqrt(eta)*np.sqrt(grad_variance[b])/fh )
    else:
      transmitt_scalar.append(0)

  return f_beam/np.sqrt(eta), np.array(transmitt_scalar)
#------------------------------------------------------------------------
def Randomization(F_SDR, num_random_samples, devices_selection_array, scaled_channel_matrix, num_of_antennas, num_of_clients):
  # This function applies the randmoization to the solution found by the SDR.
  rank= np.linalg.matrix_rank(F_SDR)
  #print( 'Ranke of the matrix is: ', rank )
  eigen_values, eigen_vectors = np.linalg.eig(F_SDR)
  #print(eigen_values)
  eigen_values= np.clip(np.real(eigen_values), 0, None)
  idx = np.argsort(eigen_values)
  eigen_values= eigen_values[idx]
  eigen_vectors= eigen_vectors[:,idx]
  if (rank == 1):
    return eigen_vectors[:, num_of_antennas-1]

  else:
    sigma= np.diag(eigen_values)
    sigma_sqrt= np.sqrt(sigma)
    randomization= np.random.normal(0, 1, (num_of_antennas, num_random_samples ))
    Candidates = (eigen_vectors@sigma_sqrt)@ randomization
    for b in range(num_random_samples):
      for i in range(num_of_clients):
        if devices_selection_array[i]==1:
          h_conj= np.conjugate(scaled_channel_matrix[:,i])
          d= np.outer(scaled_channel_matrix[:,i], h_conj)
          constraint= np.conjugate(Candidates[:,b])@d@Candidates[:,b]
          if(constraint < 1):
            Candidates[:,b] = Candidates[:,b]/np.sqrt(constraint)

    candidates_norm= np.zeros(num_random_samples)
    for n in range(num_random_samples):
      candidates_norm[n]= np.linalg.norm(Candidates[:,n])

    can_idx = np.argsort(candidates_norm)
    candidates_norm= candidates_norm[can_idx]
    Candidates= Candidates[:,can_idx]


  return Candidates[:,0], candidates_norm[0]**2 
#------------------------------------------------------------------------
def SDR(scaled_channel_matrix, devices_selection_array, num_of_antennas, num_of_clients):
  # This function computes the Semi-Definite Relaxation (SDR) solution for the single-group downlink multicast beamforming problem.
  F1= cp.Variable((num_of_antennas, num_of_antennas), hermitian = True)
  objective_function = (cp.trace(F1))/1
  constraints=[ F1 >> 0]
  for i in range(num_of_clients):
    if devices_selection_array[i] ==1:
      h_conj= np.conjugate(scaled_channel_matrix[:,i])
      d= np.outer(scaled_channel_matrix[:,i], h_conj)
      constraints.append(cp.real(cp.trace(d @ F1)) >= 1)

  prob = cp.Problem(cp.Minimize(objective_function), constraints)
  prob.solve()
  #print('Problem Status is: ',prob.status ,'|', 'Optimal value: ', prob.value ) #'\n','Optimal F_SDR: ', F_SDR.value)
  #------ Apply Randomization ------
  f_prime_sdr, loss= Randomization(F1.value, 1000 , devices_selection_array, scaled_channel_matrix, num_of_antennas, num_of_clients)
  f_sdr=  f_prime_sdr/ np.linalg.norm(f_prime_sdr,2)

  return f_sdr, f_prime_sdr
#----------------------------------------------------------------------------

def SCA(scaled_channel_matrix, devices_selection_array, num_of_antennas, num_of_clients, channel_scale):
  # This function computes the SCA solution for the single-group downlink multicast beamforming problem.
  f_sdr, f_prime_sdr= SDR(scaled_channel_matrix, devices_selection_array, num_of_antennas, num_of_clients)
  z_init= 1*f_prime_sdr
  z= 1*z_init
  num_iterations= 10
  for k in range(num_iterations):
    
    x= cp.Variable(2*num_of_antennas)
    objective_func= cp.norm(x,2)
    constraints=[]
    for i in range(num_of_clients):
      if devices_selection_array[i] ==1:
        hh= np.outer(scaled_channel_matrix[:,i], np.conjugate(scaled_channel_matrix[:,i]))
        real_p= np.real(hh@z)
        imag_p= np.imag(hh@z)
        v= np.concatenate((real_p, imag_p))
        z_conj= np.conjugate(z)
        gamma1= np.absolute(z_conj@scaled_channel_matrix[:,i])
        gamma2 = 1+ math.pow(gamma1,2)
        constraints.append( x@ v >= gamma2/2)

    prob = cp.Problem(cp.Minimize(objective_func), constraints)
    prob.solve(solver= cp.CVXOPT)
    if (abs((prob.value)**2- (np.linalg.norm(z,2))**2) < 1e-2):
      break

    z= x.value[0:num_of_antennas] + 1j* x.value[num_of_antennas:2*num_of_antennas]
  

  f_prime_sca= x.value[0:num_of_antennas] + 1j* x.value[num_of_antennas:2*num_of_antennas]
  f_sca= f_prime_sca/ np.linalg.norm(f_prime_sca,2)

  return f_sca, channel_scale*f_prime_sca ## Note: scale is multiplied to compensate for the channel scaling.  

#-------------------------------------------------------
def UpdateSpaceBasis(space, vec):
  # This function updates the basis set for a subspace (space) by adding the orthogonal component of a given vector to the set of basis.
  new_space = []
  q = 1*vec
  for i in range(len(space)):
    new_space.append(space[i])
    rij = vec@ np.conj(space[i])
    q = q - (rij*space[i])/(space[i]@np.conj(space[i]))

  if (np.linalg.norm(q) > 1e-10):
    new_space.append(q)

  return new_space
#--------------------------------------------------------
def VectorProjection(space, vec):
  projection = np.zeros(len(vec))+1j*np.zeros(len(vec))
  for i in range(len(space)):
    projection += ((vec @np.conj(space[i]))/(space[i]@np.conj(space[i])))*space[i]

  return projection
#--------------------------------------------------------
def MaxProjectionIndex(space, channel_matrix, device_selection_vector):
  # This function finds the index of the device that has the maximum channle projection norm on a space with given basis among the unchosen devices.
  projected_values= np.zeros(len(channel_matrix))
  for i in range(len(channel_matrix)):
    if device_selection_vector[i]==0:
      projected_vector= VectorProjection(space, channel_matrix[i])
      projected_values[i]= np.linalg.norm(projected_vector)
  max_indx= np.argmax(projected_values)

  return max_indx

#------------------------------------------------------------
def ObjectiveFunction(beamforming_vec, channel_matrix, devices_selection_array, noise_variance, P_zero, clients_num_sample):
  # This function returns the value for the objective function we aim to minimize 
  term1= (4/(np.sum(clients_num_sample)**2))*(((1- devices_selection_array)@clients_num_sample)**2)
  if np.sum(devices_selection_array) != 0:
    ## Coefficient 0.5 for noise varaince, is beacsue the noise distribution follows a complex symmetric distrbution.
    term2= (0.5*noise_variance/P_zero)/((devices_selection_array@clients_num_sample)**2)
    fh= channel_matrix@np.conjugate(beamforming_vec)
    term3 = (clients_num_sample**2)*devices_selection_array/(np.abs(fh)**2)
    term4= np.max(term3)
    return term1+ term2*term4
  else:
    return term1

#-------------------------------------------------------------
def GSDS(scaled_channel_matrix, channel_matrix, noise_variance, P_zero, num_of_antennas, num_of_clients, clients_num_sample):
  channel_scale = abs(clients_num_sample[0]*scaled_channel_matrix[0][0]/ channel_matrix[0][0])
  device_selection_vector= np.zeros(len(channel_matrix))
  channel_norms=[]
  Set_of_basis=[]
  information_matrix= np.zeros((num_of_clients, num_of_clients+1)) # Stores the device selection array together with corresponding objective function value.
  for i in range(len(channel_matrix)):
    channel_norms.append(np.linalg.norm(channel_matrix[i]))

  max_indx = np.argmax(channel_norms)
  device_selection_vector[max_indx] = 1
  Set_of_basis.append(channel_matrix[max_indx])

  f_bench, _ = SCA(scaled_channel_matrix, device_selection_vector, num_of_antennas, num_of_clients, channel_scale)
  information_matrix[0, 0:num_of_clients] = 1*device_selection_vector
  information_matrix[0, num_of_clients] = ObjectiveFunction(f_bench, channel_matrix, device_selection_vector, noise_variance, P_zero, clients_num_sample)

  for j in range(num_of_clients-1):
    indx= MaxProjectionIndex(Set_of_basis, channel_matrix, device_selection_vector)
    device_selection_vector[indx] = 1
    f_bench, _ = SCA(scaled_channel_matrix, device_selection_vector, num_of_antennas, num_of_clients, channel_scale)
    information_matrix[j+1, 0:num_of_clients] = 1*device_selection_vector
    information_matrix[j+1, num_of_clients] = ObjectiveFunction(f_bench, channel_matrix, device_selection_vector, noise_variance, P_zero, clients_num_sample)

    Set_of_basis = UpdateSpaceBasis(Set_of_basis , channel_matrix[indx]) # Updates the set of basis for the subspace formed by the channle vectors of the selected devices by adding the channel vector of the newly selected device

  
  indx= np.argmin(information_matrix[:, num_of_clients])
  GSDS_device_selection = information_matrix[indx, 0:num_of_clients]
  f_GSDS,_ = SCA(scaled_channel_matrix, GSDS_device_selection , num_of_antennas, num_of_clients, channel_scale)

  print('Num of selected devices by GSDS: ', np.sum(GSDS_device_selection))

  return f_GSDS, GSDS_device_selection

#----------------------------------------------------------------
def MMSE(scaled_channel_matrix, channel_matrix, grad_variance, P_zero, num_of_antennas, num_of_clients, clients_num_sample):
  channel_scale = abs(clients_num_sample[0]*scaled_channel_matrix[0][0]/ channel_matrix[0][0])
  _, f_prime =  SCA(np.sqrt(P_zero)*scaled_channel_matrix/np.sqrt(grad_variance), np.ones(num_of_clients), num_of_antennas, num_of_clients, channel_scale)
  fh= channel_matrix@np.conjugate(f_prime)
  transmitt_scalar= clients_num_sample*np.sqrt(grad_variance)/fh 

  return f_prime, transmitt_scalar





