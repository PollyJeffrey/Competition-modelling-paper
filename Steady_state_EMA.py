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

#Define the form of the block diagonal matrices used in the algorithm
def Build_QkkMinus1(QkkMinus1,k):
   for i in range(min(N1,nl-k)+1):
      QkkMinus1[i,i] = kr2*k

def Build_QkkPlus1(QkkPlus1,k):
   for i in range(min(N1,nl-(k+1))+1):
      QkkPlus1[i,i] = kf2*(nr2-k)*(nl-i-k)

def Build_Qkk(Qkk,k):
   if min(N1,nl-k) != 0:
      for i in range(1,min(N1,nl-k)):
         Qik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
         Qkk[i,i] = -Qik
         Qkk[i,i-1] = kr1*i
         Qkk[i,i+1] = kf1*(nr1-i)*(nl-i-k)
      i = 0
      Qik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
      Qkk[i,i] = -Qik
      Qkk[i,i+1] = kf1*(nr1-i)*(nl-i-k)
      i = min(N1,nl-k)
      Qik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
      Qkk[i,i] = -Qik
      Qkk[i,i-1] = kr1*i
   else:
      i = 0
      Qik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
      Qkk[0,0] = -Qik

#Function to compute the EMA steady state distribution
def EMA_steady_state_dist():
   #Compute the number of states per level L(k)
   size = []
   for k in range(N2+1):
      count=0
      for r in range(min(N1,nl-k)+1):
         count+=1
      size.append(count)
      
   #Implement Algorithm 2 for the steady state distribution
   H = [np.matrix(np.zeros((size[i],size[i]))) for i in range(N2+1)]
   invH = [ np.matrix(np.zeros((size[i],size[i]))) for i in range(N2+1)]
   k = 0
   Qkk=np.matrix(np.zeros((size[k],size[k])))
   Build_Qkk(Qkk,k)
   H[k] = Qkk
   invH[k] = (H[k]).I
   for k in range(1,N2+1):
      Qkk=np.matrix(np.zeros((size[k],size[k])))
      Build_Qkk(Qkk,k)
      QkkMinus1=np.matrix(np.zeros((size[k],size[k-1])))
      Build_QkkMinus1(QkkMinus1,k)
      QkkPlus1=np.matrix(np.zeros((size[k-1],size[k])))
      Build_QkkPlus1(QkkPlus1,k-1)
      H[k]=Qkk-QkkMinus1*invH[k-1]*QkkPlus1
      if k != N2:
         invH[k] = (H[k]).I
      else:
         zeros = np.zeros(min(N1,int(nl-N2)))
         one = np.array((1))
         f = np.hstack((one,zeros))
         H[k][:,min(N1,int(nl-N2))] = np.matrix((np.ones((len(H[k]))))).T
         invH[k] = (H[k]).I                  
   piaux=[np.matrix(np.zeros((1,size[i]))) for i in range(N2+1)]
   pi=[np.matrix(np.zeros((1,size[i]))) for i in range(N2+1)]
   
   piaux[N2] = (np.matrix((f[::-1]))*invH[N2])
   for j in reversed(range(N2)):
      QkkMinus1=np.matrix(np.zeros((size[j+1],size[j])))
      Build_QkkMinus1(QkkMinus1,j+1)
      piaux[j]=-piaux[j+1]*QkkMinus1*invH[j]
   
   tot_sum = []
   for i in range(N2+1):
      tot_sum.append(np.sum(piaux[i]))
   totsum = np.sum(tot_sum)   
   
   lengths = []
   for i in range(N2+1):
      pi[i] = piaux[i]/totsum
      length = len(pi[i].T)
      lengths.append(length)
   
   max_len = max(lengths)
   pi_full = []
   for i in range(N2+1):
      add_zeros = max_len - lengths[i]
      if add_zeros == 0:
         pi_full.append(np.asarray((pi[i])).flatten())
      else:
         pi_new = np.hstack((np.asarray((pi[i])).flatten(),np.zeros(add_zeros)))
         pi_full.append(pi_new)
   
   pi_EMA = np.stack(pi_full)
   return pi_EMA

#Compute the expected number of monomers of type 1 and 2 in steady state
def EMA_expected_m1():
   parts1 = []
   for m1 in range(N1+1):
      cols1 = []
      for m2 in range(min(N2,nl-m1)+1):
         cols1.append(pi_EMA[m2,m1])
      parts1.append(m1*np.sum(cols1))
   exp_m1 = np.sum(parts1)
   return exp_m1

def EMA_expected_m2():
   parts2 = []
   for m2 in range(N2+1):
      cols2 = []
      for m1 in range(min(N1,nl-m2)+1):
         cols2.append(pi_EMA[m2,m1])
      parts2.append(m2*np.sum(cols2))
   exp_m2 = np.sum(parts2)
   return exp_m2

#Run the code to compute the steady state distribution and expected values of m1 and m2 in steady state
pi_EMA = EMA_steady_state_dist()
exp_m1_EMA = EMA_expected_m1()
exp_m2_EMA = EMA_expected_m2()
