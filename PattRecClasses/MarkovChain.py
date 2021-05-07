import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        S = np.empty(tmax)
        S[0] = DiscreteD(self.q).rand(1)
        
        for i in range(1,tmax): # loops until tmax-1 -> Does not include the end state
            index = int(S[i-1])
            #print(index)
            #print("A test", self.A[index,:])
            value = DiscreteD(self.A[index,:]).rand(1)
            if value == self.A.shape[0]:
                #print('Finite chain has come to an end at index', i, 'instead of ', tmax) 
                return S[0:i]
            S[i] = value
        return S        

    def forward(self,pX):
        """
        [ahat, c] = forward(pX)  calculates the scaled forward variables ahat and forward scale factor ct

        Input:
        pX =    proportional to b(x) which is the state-conditional pmass/density for each state and frame
                in the observed sequence. (N,T)

        Output:  !!! NOTE that outputs will also be scaled since input is scaled !!!
        ahat =  scaled forward variable. (N,T)
        c =    forward scale factor. (1,T) for infinite and (1,T+1) for finite

        Performs the three steps of the Forward algorithm specified by the compendium on page 108:
            Initialization 
            Forward step
            Termination

        Compatible with both finite and infinite HMM.
        """
        
        T = pX.shape[1]  # extracts the length of the observed sequence

        # Variable allocation
        a_temp = np.empty([self.nStates,T])  # temporary forward variable
        ahat = np.empty([self.nStates,T])  # scaled forward varialbes 
        if self.is_finite:  # forward scale factors for finite and infinite state
            c = np.zeros(T+1)  # extra c to account for exit state
        else:
            c = np.zeros(T)  

        ### Initialization: First element (index 0) in each list
        # a_temp[:,0] calculation
        for j in range(self.nStates):
            a_temp[j,0] = self.q[j]*pX[j,0]  # b in compendium formula is substituted by pX 
        # c[0] calculation
        for index in range(a_temp[:,0].shape[0]):  # TODO perhaps a sum would be more efficient 
            c[0] = c[0] + a_temp[index,0] 
        # ahat[:,0] calculation
        ahat[:,0] = a_temp[:,0] / c[0]

        ### Forward step:
        for t in range(1,T): 
            # a_temp[:,t] calculation
            for j in range(self.nStates):
                # multiplying the two row vectors ahat[:,t-1] and A[:,j] sums them directly
                a_temp[j,t] = pX[j,t]*np.matmul(ahat[:,t-1],self.A[:,j])  
            # c[t] calculation
            c[t] = np.sum(a_temp[:,t])
            # ahat[:,t] calculation
            ahat[:,t] = a_temp[:,t] / c[t]
        
        ### Termination:
        if self.is_finite:
            c[T] = np.matmul(ahat[:,T-1],self.A[:,self.nStates])  # same here
        return [ahat, c]

    def backward(self, pX, c):
        """
        [bhat] = backward(pX, c) 
        
        Input:
        pX =    proportional to b(x) which is the state-conditional pmass/density for each state and frame
                in the observed sequence. (T,N)
        c =     forward scale factor. (1,T) for infinite and (1,T+1) for finite

        Output:
        bhat =  scaled backward variables. (N,T)
        """

        T = pX.shape[1]  # extracts the length of the observed sequence
        
        # Variable allocation
        beta = np.empty([self.nStates,T])
        bhat = np.empty([self.nStates,T])

        ### Initialization
        if self.is_finite:
            for i in self.nStates:
                beta[i,T-1]= self.A[i,self.nStates]
                bhat[i,T-1] = beta[i,T-1]/(c[T-1]*c[T])
        else:
            beta[i,T-1] = 1
            bhat[i,T-1] = 1/c[T-1]
        
        ### Backward step
        for t in range(T-1,-1,-1):
            for i in range(self.nStates):
                for j in range(self.nStates):
                    sum = self.A[i,j]*pX[t+1,j]*bhat[j,t+1]
                bhat[i,t] = 1/c[t]* sum

        return bhat
        

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def finiteDuration(self):
        pass
    
    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
