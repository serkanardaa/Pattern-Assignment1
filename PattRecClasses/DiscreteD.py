import numpy as np

class DiscreteD:
    """
    DiscreteD - class representing random discrete integer.
    
    A Random Variable with this distribution is an integer Z
    with possible values 1,...,length(ProbMass).
    
    Several DiscreteD objects may be collected in an array
    """
    def __init__(self, x):
        self.pseudoCount = 0
        self.probMass = x/np.sum(x)
        
    def rand(self, nData):
        """
        R=rand(nData) returns random scalars drawn from given Discrete Distribution.
        
        Input:
        nData= scalar defining number of wanted random data elements
        
        Result:
        R= row vector with integer random data drawn from the DiscreteD object
           (size(R)= [1, nData]
        """

        # J: First manual implementation
        """ 
        R = np.random.rand(nData) 
        # Assume probMass from init  NOT __init__
        maxVal = len(self.probMass)
        poss_val = np.arange(maxVal) # with my suggested init this has right length

        for i in range(nData):
            max_index = 0
            max_val = 0
            for j in poss_val:
                rand_probMass = self.probMass[j]*np.random.rand(1)
                if np.max(rand_probMass) > max_val:
                    max_index = j
                    max_val = np.max(rand_probMass)
            R[i] = poss_val[max_index]
            # Old uniform implementation independent of input dist
            #R[i] = np.random.rand(1)*maxVal
            #R[i] = np.round(R[i],0)
        """
        # Numpy solution
        R = np.random.choice(len(self.probMass), nData, p=self.probMass)

        return R
        
    def init(self, x):
        """
        initializes DiscreteD object or array of such objects
        to conform with a set of given observed data values.
        The agreement is crude, and should be further refined by training,
        using methods adaptStart, adaptAccum, and adaptSet.
        
        Input:
        x=     row vector with observed data samples
        
        Method:
        For a single DiscreteD object: Set ProbMass using all observations.
        For a DiscreteD array: Use all observations for each object,
               and increase probability P[X=i] in pD(i),
        This is crude, but there is no general way to determine
               how "close" observations X=m and X=n are,
               so we cannot define "clusters" in the observed data.
        """
        if len(np.shape(x))>1: 
            print('DiscreteD object can have only scalar data')
           
        x = np.round(x)
        maxObs = int(np.max(x) )
        # collect observation frequencies

        # J: I think their indexing and loop is wrong. 
        
        """
        fObs = np.zeros(maxObs) # observation frequencies
        # the list is one index too short if we want to have both max and 0
        for i in range(maxObs): 
            fObs[i] = 1 + np.sum(x==i)
            # does not take the upper bound (maxObs) into account
            # i = 0:maxObs-1
            # also 1 + makes it not work properly
        """
        
        # J: my new suggestion
        fObs = np.zeros(maxObs+1)

        for i in range(maxObs+1):
            fObs[i] = np.sum(x==i)

        # J: end of suggestion. Run test.py/discrete_pmass_test() for verification
        
        self.probMass = fObs/np.sum(fObs)

        return self


    def entropy(self):
        pass

    def prob(self):
        pass
    
    def double(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
