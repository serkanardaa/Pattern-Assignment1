from DiscreteD import DiscreteD
from MarkovChain import MarkovChain
import numpy as np
from GaussD import GaussD

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
    # Monte Carlo
    x = np.array([1,2,3,4,5,1])
    M = 1000
    #rand_samp = np.array(M)
    rand_sum = np.zeros(np.max(x)+1)
    for i in range(M):
        dist1 = DiscreteD(x)
        print("Dist1", dist1.probMass)
        #dist2 = dist1.init(x)
        #print("Dist2", dist2.probMass)
        nData = 5

        rand_samp = dist1.rand(nData)

        print(rand_samp)
        for j in range(np.max(x)+1):
            rand_sum[j] = rand_sum[j] + np.sum(rand_samp==j)
    result = rand_sum/(nData*M)
    print(result)
    print(np.sum(result))

def markov_test():
    x = np.array([1,2,3])
    d1 = DiscreteD(x)
    q = d1.probMass
    A = np.array([[1,2,3],[4,5,6],[3,4,5]])
    A = DiscreteD(A).probMass
    print(q)
    print(A[1,2])

    mc = MarkovChain(q,A)
    S = mc.rand(30)

    print(S)
    print(len(S))

def HMM_test():
    
    x = np.array([1,2,3])
    d1 = DiscreteD(x)
    q = d1.probMass
    A = np.array([[1,2,3],[4,5,6],[3,4,5]])
    A = DiscreteD(A).probMass
    print(q)
    print(A[1,2])

    mc = MarkovChain(q,A)
    S = mc.rand(30)
    
    mc = MarkovChain(q,A)
    S = mc.rand(30)

    print(S)
    print(len(S))
    
    g1 = GaussD([2])
    g2 = GaussD([3])
    hmm = HMM(mc, [g1,g2])
    

def main():
    #discrete_rand_test()
    #markov_test()
    #init_pmass_test()
    HMM_test()

if __name__ == "__main__":
    main()
