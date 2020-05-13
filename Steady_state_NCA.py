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
      
#Function to compute the NCA steady state distribution
def NCA_steady_state_dist():
   #Compute the steady state probabilities for the independent Markov processes
   SS_m1 = []
   for m1 in range(N1+1):
      p = (kf1*nl)/(kr1+(kf1*nl))
      pi_m1 = stats.binom(nr1,p).pmf(m1)
      SS_m1.append(pi_m1)
   
   SS_m2 = []
   for m2 in range(N2+1):
      p = (kf2*nl)/(kr2+(kf2*nl))
      pi_m2 = stats.binom(nr2,p).pmf(m2)
      SS_m2.append(pi_m2)

   #Compute the product of the two steady state distributions
   pi_NCA = np.empty((len(SS_m2),len(SS_m1)))
   for i in range(len(SS_m2)):
      for j in range(len(SS_m1)):
         pi_NCA[i,j] = SS_m2[i]*SS_m1[j]
   
   return pi_NCA

#Compute the expected number of monomers of type 1 and 2 in steady state
def NCA_expected_m1():
   parts1 = []
   for m1 in range(N1+1):
      cols1 = []
      for m2 in range(min(N2,nl-m1)+1):
         cols1.append(pi_NCA[m2,m1])
      parts1.append(m1*np.sum(cols1))
   exp_m1 = np.sum(parts1)
   return exp_m1

def NCA_expected_m2():
   parts2 = []
   for m2 in range(N2+1):
      cols2 = []
      for m1 in range(min(N1,nl-m2)+1):
         cols2.append(pi_NCA[m2,m1])
      parts2.append(m2*np.sum(cols2))
   exp_m2 = np.sum(parts2)
   return exp_m2

#Run the code to compute the steady state distribution and expected values of m1 and m2 in steady state
pi_NCA = NCA_steady_state_dist()
exp_m1_NCA = NCA_expected_m1()
exp_m2_NCA = NCA_expected_m2()
