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
#Number of moments to compute (l=2 will compute the mean time)
l = 2

#Define the form of the block diagonal matrices used in the algorithm
def Build_AkkMinus1(AkkMinus1,k):
   for i in range(min(N1,nl-k)+1):
      Aik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
      AkkMinus1[i,i] = (kr2*k)/Aik

def Build_AkkPlus1(AkkPlus1,k):
   for i in range(min(N1,nl-(k+1))+1):
      Aik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
      AkkPlus1[i,i] = (kf2*(nr2-k)*(nl-i-k))/Aik

def Build_Akk(Akk,k):
   for i in range(1,min(N1,nl-k)):
      Aik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
      Akk[i,i-1] = (kr1*i)/Aik
      Akk[i,i+1] = (kf1*(nr1-i)*(nl-i-k))/Aik
   i = 0
   Aik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
   Akk[i,i+1] = (kf1*(nr1-i)*(nl-i-k))/Aik
   i = min(N1,nl-k)
   Aik = kf1*(nr1-i)*(nl-i-k) + kr1*i + kf2*(nr2-k)*(nl-i-k) + kr2*k
   Akk[i,i-1] = (kr1*i)/Aik

#Function to compute the mean time to reach N monomers of type 2 from initial state (m1,m2)
def EMA_mean_time():
   #Compute the number of states per level L(k)   
   size = []
   for k in range(N2+1):
      count=0
      for r in range(min(N1,nl-k)+1):
         count+=1
      size.append(count)
         
   m = [np.matrix(np.zeros((size[i],l))) for i in range(N)]
   for k in range(N):
      m[k][:,0] = np.reshape(np.ones((size[k])),(size[k],1))
      
   b = [np.matrix(np.zeros((size[i],l))) for i in range(N)]
   
   def Build_b(b,m,j,p):
      for i in range(size[j]):
         Aij = kf1*(nr1-i)*(nl-i-j) + kr1*i + kf2*(nr2-j)*(nl-i-j) + kr2*j
         b[i][:,p] = ((p+1)*m[i][:,p])/Aij
   
   for j in range(N):
      Build_b(b[j],m[j],j,0)
         
   R = [np.matrix(np.zeros((size[i],size[i]))) for i in range(N)]
   invR = [ np.matrix(np.zeros((size[i],size[i]))) for i in range(N)]
   
   p = 0
   k = 0
   
   Akk=np.matrix(np.zeros((size[k],size[k])))
   Build_Akk(Akk,k)
   R[k] = np.identity(size[k])-Akk
   invR[k] = (R[k]).I
   
   for p in range(1,l):
      S = [np.matrix(np.zeros((size[i],1))) for i in range(N)]
      S[0] = invR[0]*b[0][:,p-1]
         
      for k in range(1,N):
         Akk=np.matrix(np.zeros((size[k],size[k])))
         Build_Akk(Akk,k)      
         AkkMinus1=np.matrix(np.zeros((size[k],size[k-1])))
         Build_AkkMinus1(AkkMinus1,k)
         AkkPlus1=np.matrix(np.zeros((size[k-1],size[k])))
         Build_AkkPlus1(AkkPlus1,k-1)
         R[k] = np.identity(size[k])-Akk-AkkMinus1*invR[k-1]*AkkPlus1
         invR[k] = (R[k]).I
         S[k] = invR[k]*AkkMinus1*S[k-1] + invR[k]*b[k][:,p-1]
   
      m[N-1][:,p] = S[N-1]
      for k in reversed(range(N-1)):
         AkkPlus1=np.matrix(np.zeros((size[k],size[k+1])))
         Build_AkkPlus1(AkkPlus1,k)      
         m[k][:,p] = S[k] + invR[k]*AkkPlus1*m[k+1][:,p]
   
      for k in range(N):
         Build_b(b[k],m[k],k,p)         
      
   return m

#Run the code to compute the mean time to reach N monomers of type 2 from initial state (m1,m2)=(0,0)
Time_EMA = EMA_mean_time()[0][0,1]
