import numpy as np

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

#Set number of monomers of type 2 to reach
N = 10

#Function to compute the mean time to reach N monomers of type 2 from initial state 0
def NCA_mean_time():
   #Define all of the rates of the 1D Markov process
   lambda_i = []
   mu_i = []
   for i in range(N+1):
      lambda_f = kf2*(nr2-i)*(nl-i)
      lambda_i.append(lambda_f)
      mu_b = kr2*i
      mu_i.append(mu_b)

   #Compute the mean time to reach N monomers from initial state 0
   times = []
   T0 = 1/lambda_i[0]
   times.append(T0)
   for i in range(1,N):
      Ti = 1/lambda_i[i] + (mu_i[i+1]/lambda_i[i])*times[i-1]
      times.append(Ti)

   mean_time = np.sum(times)         
   return mean_time

#Run the code to compute the mean time to reach N monomers of type 2 from initial state (m1,m2)=(0,0)
Time_NCA = NCA_mean_time()
