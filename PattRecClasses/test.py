from DiscreteD import DiscreteD
#from MarkovChain import MarkovChain
import numpy as np

def markov_test():
    """
    d1 = DiscreteD(x)
    d1.init()
    mc = MarkovChain()
    x = 10 # placeholder
    """
    pass

def init_pmass_test():
    weight = 5
    x = np.random.rand(10)*weight
    x = np.round(x)
    maxVal = np.max(x) 
    poss_val = np.arange(maxVal)
    print(poss_val)
    print("END") 
    sum = np.sum(x==0)
    print(x)
    print(sum)
def discrete_pmass_test():
    print("START TEST")
    
    x = np.array([1,2,3,4,5,1])
    dist1 = DiscreteD(x)
    # print("Dist1", dist1.probMass)
    dist2 = dist1.init(x)
    print("Dist2", dist2.probMass)


def discrete_rand_implement_test():
    x = np.array([1,2,3,4,5,1])
    dist1 = DiscreteD(x)
    # print("Dist1", dist1.probMass)
    dist2 = dist1.init(x)
    print("Dist2", dist2.probMass)
    nData = 5

    print("START TEST")
    maxVal = np.max(x) 
    poss_val = np.arange(maxVal+1)
    print(poss_val)
    R = np.random.rand(nData)
    for j in range(nData):
        max_index = 0
        max_val = 0
        for i in poss_val:
            r_pmass = dist2.probMass[i]*np.random.rand(1)
            print(r_pmass)
            if np.max(r_pmass) > max_val:
                max_index = i
                max_val = np.max(r_pmass)
                print(max_index)
        R[j] = poss_val[max_index]        

    print(R)
    #ran1 = dist1.rand(5)
    #ran2 = dist2.rand(5)
    # print(ran1)

def discrete_rand_test():
    x = np.array([1,2,3,4,5,1])
    dist1 = DiscreteD(x)
    # print("Dist1", dist1.probMass)
    dist2 = dist1.init(x)
    print("Dist2", dist2.probMass)
    nData = 5

    rand_samp = dist2.rand(nData)

    print(rand_samp)

def main():
    discrete_rand_test()
    #markov_test()
    #init_pmass_test()
    pass

if __name__ == "__main__":
    main()