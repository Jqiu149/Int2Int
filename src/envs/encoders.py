from abc import ABC, abstractmethod
import numpy as np
import math
class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, val):
        pass
   
    def decode(self, lst):
        v, p = self.parse(lst)
        if p == 0:
            return None
        return v

class SymbolicInts(Encoder):
    """
    one token per int from min to max (0 to 1 for binary, -10 to 10 for bounded ints, 0 to Q-1 for modular)
    optionally: add a prefix, e.g. E-100 E100 for exponents, N1 N5 for dimensions 
    """
    def __init__(self, min, max, prefix = ''):
        super().__init__()
        self.prefix = prefix
        self.symbols = [self.prefix + str(i) for i in range(min, max+1)]

    def encode(self, value):
        return [self.prefix+str(value)]

    def parse(self, lst):
        if len(lst) == 0 or (not lst[0] in self.symbols):
            return None, 0
        return  int(lst[0][len(self.prefix):]), 1


class PositionalInts(Encoder):
    """
    Single integers, in base params.base (positive base), with the sign
    """
    def __init__(self, base=10):
        super().__init__()
        self.base = base
        self.symbols = ['+', '-'] + [str(i) for i in range(self.base)]

    def encode(self, value):
        if value != 0:
            prefix = []
            w = abs(value)
            while w > 0:
                prefix.append(str(w % self.base))
                w = w // self.base
            prefix = prefix[::-1]
        else:
            prefix =['0']
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def parse(self,lst):
        if len(lst) <= 1 or (lst[0] != '+' and lst[0] != '-'):
            return None, 0
        res = 0
        pos = 1
        for x in lst[1:]:
            if not (x.isdigit()):
                break
            res = res * self.base + int(x)
            pos += 1
        if pos < 2: return None, pos
        return -res if lst[0] == '-' else res, pos

class PositionalIntsRev(Encoder):
    """
    Single integers, in base params.base (positive base), with the sign
    """
    def __init__(self, base=10):
        super().__init__()
        self.base = base
        self.symbols = ['+', '-'] + [str(i) for i in range(self.base)]

    def encode(self, value):
        if value != 0:
            prefix = []
            w = abs(value)
            while w > 0:
                prefix.append(str(w % self.base))
                w = w // self.base
            
        else:
            prefix =['0']
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def parse(self,lst):
        if len(lst) <= 1 or (lst[0] != '+' and lst[0] != '-'):
            return None, 0
        res = 0
        pos = 1
        for x in reversed(lst[1:]):
            if not (x.isdigit()):
                break
            res = res * self.base + int(x)
            pos += 1
        if pos < 2: return None, pos
        return -res if lst[0] == '-' else res, pos


def max_fit_exp(num, base):
    assert num>=0, "max_fit_exp expects a non-negative integer for num"
    assert base>=2, "max_fit_exp expects base to be >=2"
    
    max_exp= 0;
    num = num// base
    while( num>0):
       max_exp =max_exp + 1
       num = num//base

    return max_exp

# encode integers in base given with token for exp size after each digit
# i.e encode 432 in base 10 as 4 e2 3 e1 2 e0
class PositionalIntsModified(Encoder):
    """
    Single integers, in base params.base (positive base), with the sign
    """
    def __init__(self, max_abs_int, base=10):
        super().__init__()
        self.base = base
        self.symbols = ['+', '-'] + [str(i) for i in range(self.base)]        
        self.symbols = self.symbols + ["e" + str(i) for i in range (max_fit_exp(max_abs_int, base)+1)]

    def encode(self, value):
        if value != 0:
            prefix = []
            w = abs(value)
            i = 0
            while w > 0:
                prefix.append( "e" + str(i))
                prefix.append(str(w % self.base))
                i=i+1
                w = w // self.base
            prefix = prefix[::-1]
        else:
            prefix =['0', "e0"]
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def parse(self,lst):
        if len(lst) <= 1 or (lst[0] != '+' and lst[0] != '-'):
            return None, 0
        res = 0
        pos = 1
        for x in lst[1::2]:
            if not (x.isdigit()):
                break
            res = res * self.base + int(x)
            pos += 1
        if pos < 2: return None, pos
        return -res if lst[0] == '-' else res, pos

class PositionalIntsExp(Encoder):
    """
    Single integers, in base params.base (positive base), with the sign
    """
    def __init__(self, max_abs_int, base=10, reverse= False):
        super().__init__()
        self.base = base
        self.reverse = False
        self.symbols = ['+', '-', 'e'] + [str(i) for i in range(max(max_fit_exp(max_abs_int, self.base)+1, self.base))]        

    def encode(self, value):
        if value != 0:
            prefix = []
            w = abs(value)
            i = 0
            while w > 0:
                prefix.append (str(i))
                prefix.append( "e")
                prefix.append(str(w % self.base))
                i=i+1
                w = w // self.base
            if( not self.reverse):
                prefix = prefix[::-1]

        else:
            prefix =['0', "e0"]
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def parse(self,lst):
        if len(lst) <= 1 or (lst[0] != '+' and lst[0] != '-'):
            return None, 0
        res = 0
        pos = 1
        start = -1 if self.reverse else 1

        for x in lst[start::3]:
            if not (x.isdigit()):
                break
            res = res * self.base + int(x)
            pos += 1
        if pos < 2: return None, pos
        return -res if lst[0] == '-' else res, pos

class PositionalIntsPaired(Encoder):
    """
    expected input: list of 2 integers x1x2...xn and y1...ynwith signs s1 and s2 
        ... note some of xi yi might be zero even if leading with them
    output form: (s1 x1 s2 y1) ..... (s1 x1 s2 y2)
        or same thing withotu the s1, s2 if includeSigns is false
    """

    def __init__(self, base=10, includeSigns = True, reverseOrder = False):
        super().__init__()
        self.base = base
        self.symbols = ['+', '-'] + [str(i) for i in range(self.base)]
        self.includeSigns = includeSigns
        self.reverseOrder = reverseOrder

    def encode(self, inputs): 
        assert np.shape(inputs) == (2,), "inputs is expected to be an np array of length 2)"

        w1 = abs( inputs[0]);
        w2 = abs(inputs[1]);
        s1 = "+" if inputs[0]>= 0 else "-" 
        s2= "+" if inputs[1]>= 0 else "-" 


        prefix = []

        if self.includeSigns:
            if(w1 ==0 and w2 ==0):
                prefix.append(["(", "+", "0", "+", "0", ")"])
            
            while (w1 > 0 or w2 >0):
                prefix.append([ "(", s1, str(w1 % self.base), s2,str(w2 % self.base), ")"] )
                w1 = w1 //self.base
                w2 = w2 // self.base
        else:
            if(w1 ==0 and w2 ==0):
                prefix.append( ["(", "0","0", ")"])

            while (w1 > 0 or w2 >0):
                prefix.append([ "(", str(w1 % self.base), str(w2 % self.base), ")"])
                w1 = w1 //self.base
                w2 = w2 // self.base

        if (not self.reverseOrder):
            prefix.reverse()
            
        return [x for xs in prefix for x in xs]


    def parse(self,lst):


        if len(lst) < 4:
            return None, 0
        spacing = 6 if self.includeSigns else 4
        start1 = 2 if self.includeSigns else 1
        start2 = 4 if self.includeSigns else  2


        res1 = 0
        res2 = 0

        for pos, x in enumerate(lst[start1::spacing]):
            if not (x.isdigit()):
                return 0
            if( self.reverseOrder):
                res1+= int(x)* self.base**(pos)
            else:
                res1 = res1 * self.base + int(x)


        for pos, x in enumerate(lst[start2::spacing]):
            if not (x.isdigit()):
                return 0
            if (self.reverseOrder):
                res2=res2+ int(x)*self.base**pos
            else:
                res2 = res2 * self.base + int(x)


        if(self.includeSigns):
            if(lst[1] == '-'):
                res1 = -res1
            if(lst[3] == '-'):
                res2 == -res2

        return [res1, res2], 1


class PositionalIntsPairedPadded(Encoder):
    """
    expected input: list of 2 integers x1x2...xn and y1...ynwith signs s1 and s2 
        ... note some of xi yi might be zero even if leading with them
    output form: (0 0) .... (0 0) (s1 x1 s2 y1) ..... (s1 x1 s2 y2)
        or same thing withotu the s1, s2 if includeSigns is false
    """

    def __init__(self, maxLen, base=10, includeSigns = True, reverseOrder = False):
        super().__init__()
        self.base = base
        self.maxLen = maxLen
        self.symbols = ['+', '-'] + [str(i) for i in range(self.base)]
        self.includeSigns = includeSigns
        self.reverseOrder = reverseOrder

    def encode(self, inputs): 
        assert np.shape(inputs) == (2,), "inputs is expected to be an np array of length 2)"
        
        w1 = abs( inputs[0]);
        w2 = abs(inputs[1]);
        s1 = "+" if inputs[0]>= 0 else "-" 
        s2= "+" if inputs[1]>= 0 else "-" 

        #print("hii???")

        prefix = []

        currentLen= 0

        if self.includeSigns:
            if(w1 ==0 and w2 ==0):
                prefix.append(["(", "+", "0", "+", "0", ")"])
                currentLen=  6
            
            while (w1 > 0 or w2 >0):
                prefix.append([ "(", s1, str(w1 % self.base), s2,str(w2 % self.base), ")"] )
                w1 = w1 //self.base
                w2 = w2 // self.base
                currentLen+=6

                #print("loop1")

            for i in range( int( (self.maxLen -2 - currentLen )/ 6 )):
                prefix.append(["(", s1, "0", s2, "0", ")"])

                #print("loop2")

        else:
            if(w1 ==0 and w2 ==0):
                prefix.append( ["(", "0","0", ")"])
                currentLen= 4
            
            while (w1 > 0 or w2 >0):
                prefix.append([ "(", str(w1 % self.base), str(w2 % self.base), ")"])
                w1 = w1 //self.base
                w2 = w2 // self.base
                currentLen+=4

                #print("loop3")
            
            for i in range( int( (self.maxLen-2- currentLen )/ 4)):
                prefix.append(["(", "0", "0", ")"])

                #print("loop4", i, int( (self.maxLen-currentLen)/4))


        if (not self.reverseOrder):
            prefix.reverse()
            
        return [x for xs in prefix for x in xs]


    def parse(self,lst):


        if len(lst) < 4:
            return None, 0
        spacing = 6 if self.includeSigns else 4
        start1 = 2 if self.includeSigns else 1
        start2 = 4 if self.includeSigns else  2


        res1 = 0
        res2 = 0

        for pos, x in enumerate(lst[start1::spacing]):
            if not (x.isdigit()):
                return 0
            if( self.reverseOrder):
                res1+= int(x)* self.base**(pos)
            else:
                res1 = res1 * self.base + int(x)


        for pos, x in enumerate(lst[start2::spacing]):
            if not (x.isdigit()):
                return 0
            if (self.reverseOrder):
                res2=res2+ int(x)*self.base**pos
            else:
                res2 = res2 * self.base + int(x)


        if(self.includeSigns):
            if(lst[1] == '-'):
                res1 = -res1
            if(lst[3] == '-'):
                res2 == -res2

        return [res1, res2], 1




class NumberArray(Encoder):
    """
    Array of integers, in base params.base (any shape)
    TODO modify to support float, complex (rationals), different subencoders
    """
    def __init__(self, params, max_dim, dim_prefix, tensor_dim, code='pos_int'):
        super().__init__()
        self.tensor_dim = tensor_dim
        self.symbols = []
        self.dimencoder = SymbolicInts(1, max_dim, dim_prefix)
        self.symbols.extend(self.dimencoder.symbols)
        if code == 'pos_int':
            self.subencoder = PositionalInts(params.base)
        elif code =="pos_int_modified":
            self.subencoder= PositionalIntsModified(max(abs(params.minint), params.maxint), params.base)
        elif code =="pos_int_modified2":
            self.subencoder= PositionalIntsModified2(max(abs(params.minint), params.maxint), params.base)
        else:
            self.subencoder = SymbolicInts(params.minint, params.maxint)
        self.symbols.extend(self.subencoder.symbols)

        

    def encode(self, vector):
        lst = []
        assert len(np.shape(vector)) == self.tensor_dim
        for d in np.shape(vector):
            lst.extend(self.dimencoder.encode(d))
        for val in np.nditer(np.array(vector)):
            lst.extend(self.subencoder.encode(val))
        return lst

    def decode(self, lst):
        shap = [] 
        h = lst
        for _ in range(self.tensor_dim):
            v, _ = self.dimencoder.parse(h)
            if v is None:
                return None
            shap.append(v)
            h = h[1:]
        m = np.zeros(tuple(shap), dtype=int)
        for val in np.nditer(m, op_flags=['readwrite']):
            v, pos = self.subencoder.parse(h)
            if v is None:
                return None
            h = h[pos:]
            val[...] = v      
        return m
