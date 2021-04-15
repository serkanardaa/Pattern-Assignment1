from DiscreteD import DiscreteD
from MarkovChain import MarkovChain
import numpy as np

def discrete_test():

    """
    x = np.random.rand(10,1)
    print(x)
    dist = DiscreteD(x)
    print(dist.probMass)
    """
    
    # dist.init(x)
    x = np.random.rand(10,1)
    dist = DiscreteD(x)
    print(len(dist.probMass))

    y = np.random.rand(1)
    print(y)

    print("START TEST")
    ran = dist.rand(5)
    print(ran)


def markov_test():
    
    d1 = DiscreteD(x)
    d1.init()
    mc = MarkovChain()

def main():
    discrete_test()

    #markov_test()

if __name__ == "main":
    main()