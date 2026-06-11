
from abc import ABC, abstractmethod
import numpy as np
import math
from logging import getLogger

logger = getLogger()


class Generator(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

# empty for now
class Sequence(Generator):
    def __init__(self, params, dims):
        super().__init__(params)

        self.operation = params.operation
        self.maxint = params.maxint
        self.minint = params.minint
        self.dims = dims
        self.modulus = params.modulus

    # integers from 1 to maxint, log uniform distribution
    def integer_loguniform_sequence(self, len, rng, type=None, max=None):
        maxint = self.maxint if max is None else max
        lgs = math.log10(maxint)*rng.rand(len)
        return np.int64(10 ** lgs)


    # integers from minint to maxint, uniform distribution
    def integer_sequence(self, len, rng, type=None, max=None):
        maxint = self.maxint if max is None else max
        return rng.randint(self. minint, maxint + 1, len)

    # integer (n,p) matrix, uniformly distributed coefficients between -maxint and maxint 
    def integer_matrix(self, n, p, rng):
        maxint = (int)(self.maxint + 0.5)
        return rng.randint(- maxint, maxint + 1, (n, p))

    def generate(self, rng, type=None):
        if self.operation in ["fraction_simplify","fraction_round"]:
            integers = self.integer_sequence(3, rng)
            if self.operation == "fraction_simplify":
                g = math.gcd(integers[1],integers[2])
                if integers[0] == 1:
                    integers[0] = rng.randint(2, self.maxint + 1)
                inp = [integers[0] * integers[1] // g, integers[0] * integers[2] // g ]
                out = [integers[1] // g , integers[2] // g]
            else:
                m1 = min(integers[1],integers[2])
                m2 = max(integers[1],integers[2])
                if m2 == m1:
                    m1 = m2 - 1
                inp = [integers[0] * m2 + m1, m2]
                out = integers[0]
            return inp, out

        if self.operation in ["fraction_add", "fraction_compare", "fraction_determinant", "fraction_product"]:
            inp = self.integer_sequence(4, rng)
            if self.operation == "fraction_add":
                num = inp[0] * inp[3] + inp[1] * inp[2]
                den = inp[1] * inp[3]
                g = math.gcd(num, den)
                out = [int(num // g), int(den // g)]
            elif self.operation == "fraction_product":
                num = inp[0] * inp[2]
                den = inp[1] * inp[3]
                g = math.gcd(num, den)
                out = [int(num // g), int(den // g)]
            elif self.operation == "fraction_determinant":
                out = inp[0] * inp[3] - inp[1] * inp[2]    
            else: 
                cmp = inp[0] * inp[3] - inp[1] * inp[2]
                out = 1 if cmp > 0 else 0
            return inp, out
        if self.operation in ["modular_add","modular_mul"]:
                inp = self.integer_sequence(2, rng, type)
                out = (inp[0] + inp[1]) % self.modulus if self.operation =="modular_add" else (inp[0] * inp[1]) % self.modulus
                return inp, out
        if self.operation in ["gcd"]:
            inp = self.integer_sequence(2, rng, type)
            out = math.gcd(inp[0], inp[1])
            return inp, out
        if self.operation == "matrix_rank":
            maxrank = min(self.dims[0], self.dims[1])
            rank = rng.randint(1, maxrank + 1)
            
            P = self.integer_matrix(self.dims[0], rank, rng)
            Q = self.integer_matrix(rank, self.dims[1], rng)
            input = P @ Q
            check_rank = np.linalg.matrix_rank(input)
            if check_rank != rank:
                return None
            return input, rank

        return None

    def evaluate(self, src, tgt, hyp):
                        
        return 0, [],[]


class lcmGenerator(Sequence):
    def generate (self, rng, type):
        inp = self.integer_sequence(2, rng, type)
        out = math.lcm( inp[0], inp[1])
        return inp, out

class multGenerator(Sequence):
    def generate (self, rng, type2):
        inp = self.integer_sequence(2, rng, type2)
        out = inp[0]*inp[1]
        return inp, out


class multGeneratorLogUniform(Sequence):
    def generate (self, rng, type2):
        inp = self.integer_loguniform_sequence(2, rng, type2)
        out = inp[0]*inp[1]
        return inp, out

class multGenerator_1xn_LogUniform(Sequence):
    def generate (self, rng, type2):
        inp = [self.integer_loguniform_sequence(1, rng, type2)[0],  self.integer_sequence(1,rng,type2, max=9)[0]]
        out = inp[0]*inp[1]
        return inp, out

class multGenerator_1xnWeighted_LogUniform(Sequence):
    def generate (self, rng, type2, weight = 0.3):
        rand = self.integer_sequence(1,rng,type2, max=10)[0] 

        if (rand <=3 ):
            inp = [self.integer_loguniform_sequence(1, rng, type2)[0], self.integer_sequence(1,rng,type2, max=9)[0]]
        else:
            inp = [x+9 for x in self.integer_loguniform_sequence(2, rng, type2, max=self.maxint-9)]

        out = inp[0]*inp[1]
        return inp, out


class addGenerator(Sequence):
    def generate (self, rng, type2):
        inp = self.integer_sequence(2, rng, type2)
        out = inp[0]+inp[1]
        return inp, out

class addGeneratorLogUniform(Sequence):
    def generate (self, rng, type2):
        inp = self.integer_loguniform_sequence(2, rng, type2)
        out = inp[0]+inp[1]
        return inp, out

#base 10 specifically
def sumGetCarry(n1, n2):
 
    carrySum = 0
    exp = 1
    while ( n1>0 and n2>0):
        d1 = n1 % 10
        d2 = n2 % 10

        if(d1+d2 >= 10):
            carrySum = carrySum + 10**exp

        n1 = n1 // 10
        n2 = n2 // 10
        exp = exp+1

    return carrySum


# i think need to assume the base is specific thing cus.... we don't get acess to it here and the carries depend on them..
class addGeneratorStepsLogUniform(Sequence):
    
    def generate (self, rng, type2):
        inp = self.integer_loguniform_sequence(2, rng, type2)
        
        carry = sumGetCarry(inp[0], inp[1])
        out = [inp[0]+inp[1] - carry, carry]
        return inp, out


# feel like this isn't great.... some worry about like rounding and computer math in general...
# but i think we're doing integers only so might be fine? tried to avoid division... 
def pairVectorsR2LinearIndep(v1,v2):
    return  v1[0]*v2[1] != v2[0]*v1[1]

#take in 2 LINEARLY INDEPENDENT vectors in R^2 (2 lists of length 2)
# output [u1, u2] (as a sequence of 4 vectors [u11, u12, u21,u22]) where ||u1||<=||u2|| where hopefully ||u1|| is minimal in lattice?
def LagrangeReduce(v1,v2):
    assert np.shape(v1) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
    assert np.shape(v2) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
    assert pairVectorsR2LinearIndep(v1,v2), f"LagrangeReduce expects v1 and v2 to be linearly independent. v1 is ${v1} and v2 is ${v2}"

    # just for concenience for element wise operation notation... probably easier ways to do this
    v1 = np.array(v1)
    v2 = np.array(v2)

    norm1Squared = np.dot(v1, v1) 
    norm2Squared = np.dot(v2,v2) 

    done = False
    while(not done):
        if(norm1Squared> norm2Squared):
            v1,v2 = v2,v1
            norm1Squared,norm2Squared = norm2Squared,norm1Squared

        u = round( (np.dot(v1,v2))/ norm1Squared)
        v2 = v2-u*v1
        norm2Squared= np.dot(v2,v2) 

        if(norm1Squared<= norm2Squared):
            done = True

    return v1.tolist()+ v2.tolist()


#take in 2 LINEARLY INDEPENDENT vectors in R^2 (2 lists of length 2)
def LagrangeReduceOneStep(v1,v2):
    assert np.shape(v1) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
    assert np.shape(v2) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
    assert pairVectorsR2LinearIndep(v1,v2), f"LagrangeReduce expects v1 and v2 to be linearly independent. v1 is ${v1} and v2 is ${v2}"

    # just for concenience for element wise operation notation... probably easier ways to do this
    v1 = np.array(v1)
    v2 = np.array(v2)

    norm1Squared = np.dot(v1, v1) 
    norm2Squared = np.dot(v2,v2) 

    if(norm1Squared> norm2Squared):
        v1,v2 = v2,v1
        norm1Squared,norm2Squared = norm2Squared,norm1Squared

    u = round( (np.dot(v1,v2))/ norm1Squared)
    v2 = v2-u*v1
    norm2Squared= np.dot(v2,v2) 


    return v1.tolist()+ v2.tolist()




#note we're only doing vectors in Z^2. values are based on min and maxint. 
#if we want, we can later try to think about how to get more genreal things but... :/
class latticeGenerator(Sequence):
    def generate(self, rng, type2): 
        inp = self.integer_sequence(4,rng,type2)
        if inp[0] ==0 and inp[1] ==0 :
            inp[0]=1
        if inp[2] ==0 and inp[3] ==0: 
            inp[3] =1

        counter = 1
        while( not pairVectorsR2LinearIndep(inp[0:2], inp[2:4])):
            inp[2:4] = self.integer_sequence(2,rng,type2)
            counter+=1

            if(counter >100):
                raise Exception(f"okay we generated more than 100 lineraly dependent vectors in a row, something is probably wrong, vector array is: ${inp}, minInt is ${self.minit}, maxInt is ${self.maxint}")
                break
            
        out = LagrangeReduce(inp[0:2], inp[2:4])

        return inp, out


class latticeOneStepGenerator(Sequence):
    def generate(self, rng, type2): 
        inp = self.integer_sequence(4,rng,type2)
        if inp[0] ==0 and inp[1] ==0 :
            inp[0]=1
        if inp[2] ==0 and inp[3] ==0: 
            inp[3] =1

        counter = 1
        while( not pairVectorsR2LinearIndep(inp[0:2], inp[2:4])):
            inp[2:4] = self.integer_sequence(2,rng,type2)
            counter+=1

            if(counter >100):
                raise Exception(f"okay we generated more than 100 lineraly dependent vectors in a row, something is probably wrong, vector array is: ${inp}, minInt is ${self.minit}, maxInt is ${self.maxint}")
                break
            
        out = LagrangeReduceOneStep(inp[0:2], inp[2:4])

        return inp, out
