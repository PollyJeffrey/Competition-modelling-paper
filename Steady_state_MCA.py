import numpy as np
from scipy import stats

#Define number of molecules and parameter values for the model
nl = 500
nr1 = 30
nr2 = 50
kr1 = pow(10,-3)
kr2 = pow(10,-3)
kd1 = 100
kd2 = 100
kf1 = kr1/kd1
kf2 = kr1/kd2   

N1 = min(nr1,nl)
N2 = min(nr2,nl)

#Set the epsilon threshold and the value of alpha (alpha should be less than 1 if nl<nr1+nr2)
epsilon = pow(10,-5)
alpha = 1

#Function to compute the MCA steady state distribution
def MCA_steady_state_dist():
   #Initiate the list of nl values for each iteration
   nl_values = [nl]
   
   #Compute the first expected values using nli = nl
   expected_m1 = (kf1*nr1*nl_values[0])/(kr1+kf1*nl_values[0])
   expected_m2 = (kf2*nr2*nl_values[0])/(kr2+kf2*nl_values[0])
   
   expected_m1_values = [expected_m1]
   expected_m2_values = [expected_m2]
   
   nl_values = [nl]
   
   #Run Algorithm 1
   diff_m1 = 1
   diff_m2 = 1
   while abs(diff_m1) > epsilon or abs(diff_m2) > epsilon:
      nl_values.append((nl - expected_m1_values[-1] - expected_m2_values[-1])*alpha + nl_values[-1]*(1-alpha))
   
      expected_m1_i = (kf1*nr1*nl_values[-1])/(kr1+kf1*nl_values[-1])
      expected_m2_i = (kf2*nr2*nl_values[-1])/(kr2+kf2*nl_values[-1])
   
      diff_m1 = expected_m1_i-expected_m1_values[-1]
      diff_m2 = expected_m2_i-expected_m2_values[-1]
   
      expected_m1_values.append(expected_m1_i)
      expected_m2_values.append(expected_m2_i)
   
   #Compute the steady state probabilities for the two MCs using the effective value of nl
   SS_m1_MCA = []
   for m1 in range(N1+1):
      p = (kf1*nl_values[-1])/(kr1+(kf1*nl_values[-1]))
      pi_m1 = stats.binom(nr1,p).pmf(m1)
      SS_m1_MCA.append(pi_m1)
   
   SS_m2_MCA = []
   for m2 in range(N2+1):
      p = (kf2*nl_values[-1])/(kr2+(kf2*nl_values[-1]))
      pi_m2 = stats.binom(nr2,p).pmf(m2)
      SS_m2_MCA.append(pi_m2)
   
   pi_mca_1 = np.empty((len(SS_m2_MCA),len(SS_m1_MCA)))
   for i in range(len(SS_m2_MCA)):
      for j in range(len(SS_m1_MCA)):
         pi_mca_1[i,j] = SS_m2_MCA[i]*SS_m1_MCA[j]
   
   #Set to zero the probabilities for any states not in the state space
   for i in range(N2+1):
      for j in range(N1+1):
         if i + j  > nl:
            pi_mca_1[i,j] = 0
   
   #Normalise to compute the final steady state distribution computed via the MCA
   pi_MCA = pi_mca_1/np.sum(pi_mca_1)
   return pi_MCA

#Compute the expected number of monomers of type 1 and 2 in steady state
def MCA_expected_m1():
   parts1 = []
   for m1 in range(N1+1):
      cols1 = []
      for m2 in range(min(N2,nl-m1)+1):
         cols1.append(pi_MCA[m2,m1])
      parts1.append(m1*np.sum(cols1))
   exp_m1 = np.sum(parts1)
   return exp_m1

def MCA_expected_m2():
   parts2 = []
   for m2 in range(N2+1):
      cols2 = []
      for m1 in range(min(N1,nl-m2)+1):
         cols2.append(pi_MCA[m2,m1])
      parts2.append(m2*np.sum(cols2))
   exp_m2 = np.sum(parts2)
   return exp_m2

#Run the code to compute the steady state distribution and expected values of m1 and m2 in steady state
pi_MCA = MCA_steady_state_dist()
exp_m1_MCA = MCA_expected_m1()
exp_m2_MCA = MCA_expected_m2()
