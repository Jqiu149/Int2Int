from abc import ABC, abstractmethod
import numpy as np
import math
import random 


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



# the way we wrote this assumes base 10
# should probbaly make it more general .... at least for powers of 10
class PositionalRealTruncated(PositionalInts):
    def __init__(self, base = 10, numDecimalPlaces=None):
        super().__init__(base)
        self.symbols+=["."]
        self.numDecimalPlaces = int(numDecimalPlaces)

    def encode(self, val):
        
        valToStr = str(val).split('.')
        intPart = valToStr[0]
        decPart =  None if (len(valToStr) == 1 or valToStr[1]=='0')  else valToStr[1]

        res = super().encode(int(intPart))


        if(decPart != None):
            numLeadingZeros = 0 
            while(numLeadingZeros < self.numDecimalPlaces and decPart[numLeadingZeros] == '0'):
                numLeadingZeros += 1


            
            if self.numDecimalPlaces is None:
                actualDecPart = decPart[0:self.numDecimalPlaces].rstrip('0')
                if actualDecPart !='0':
                    res += ['.'] + ['0']*numLeadingZeros + super().encode(int(actualDecPart))[1:None]
            elif self.numDecimalPlaces > numLeadingZeros:
                actualDecPart = decPart[0:self.numDecimalPlaces-numLeadingZeros+1].rstrip('0')
                if actualDecPart!= '':
                    res += ['.'] + ['0']*numLeadingZeros + super().encode(int(actualDecPart))[1:None]

        return res


    def parse(self, lst):

            endOfIntPart = None
            decimalIndex = None
            for i,j in enumerate(lst[1:None]):
                if j == '+' or j== '-':
                    decimalIndex = None
                    endOfIntPart = i+1
                    break;
                if j == '.':
                    decimalIndex = i+1
                    endOfIntPart = i+1
                    break;

            leadingZeroCount = 0
            if decimalIndex is not None: 
                while lst[decimalIndex + 1 + leadingZeroCount]== '0':
                    leadingZeroCount += 1

            resInt = super().parse(lst[0:endOfIntPart])
            resDec = None if decimalIndex is None else super().parse([lst[0]]+ lst[decimalIndex+1:None]) 

#we don't need to add 1 for the . b/c the resDec[1] counds 1 for the sign we addded for it...
            pos = resInt[1] if resDec == None else resInt[1]+resDec[1]
            

            if lst[0] == '-':
                val =  int(resInt[0]) if resDec is None else float(str(resInt[0]) + "." +"0"*leadingZeroCount+ str(resDec[0])[1:None])
            else: 
              val =  int(resInt[0]) if resDec is None else float(str(resInt[0]) + "." + "0"*leadingZeroCount + str(resDec[0]))
            return val, pos


















#inp is a list
#insert period is how many list entries are betwen the positions we can insert the X char
def insertRandomX(inp, insertPeriod, minSpaces=0, maxSpaces=10):
    assert type(inp) == list
    assert type(minSpaces) == int and type(maxSpaces) ==int and type(insertPeriod) == int 
    res = []
    numInserts = len(inp)/insertPeriod
    #b/c we don't want to insert at the end i guess
    numInserts = int(numInserts-1) if numInserts.is_integer() else int(numInserts)

    
    for i in range(numInserts):
        res+=inp[i*insertPeriod:i*insertPeriod+insertPeriod]
        randomSpaces = ["X"] * random.randrange(minSpaces, maxSpaces)
        res+=randomSpaces

    res+=inp[(numInserts)* insertPeriod: ]
 
    return res

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
                return lst,0
            if( self.reverseOrder):
                res1+= int(x)* self.base**(pos)
            else:
                res1 = res1 * self.base + int(x)


        for pos, x in enumerate(lst[start2::spacing]):
            if not (x.isdigit()):
                return lst, 0
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
   


class PositionalIntsPairedAddedSpaces(PositionalIntsPaired):
    def __init__(self, base=10, includeSigns = True, reverseOrder = False):
        super().__init__(base, includeSigns, reverseOrder)
        self.symbols.append("X")

    def encode(self, inputs):
        originalRes = super().encode(inputs)
        
        digitRepLength = 6 if self.includeSigns else 4
        numDigits = len(originalRes)/digitRepLength

        return insertRandomX(originalRes, digitRepLength, 0, 7) 

    def parse(self, lst):
        return super().parse([x  for x in lst if x!= "X"])

class PositionalIntsPairedPadded(PositionalIntsPaired):
    """
    expected input: list of 2 integers x1x2...xn and y1...ynwith signs s1 and s2 
        ... note some of xi yi might be zero even if leading with them
    output form: (0 0) .... (0 0) (s1 x1 s2 y1) ..... (s1 x1 s2 y2)
        or same thing withotu the s1, s2 if includeSigns is false
    """

    def __init__(self, totalDigits, base=10, includeSigns = True, reverseOrder = False):
        super().__init__(base, includeSigns, reverseOrder)
        self.totalDigits = totalDigits

    def encode(self, inputs): 
        res = super().encode(inputs)
 
        if self.includeSigns:
            padding =  "(+0+0)"
            currentDigits = len(res)/6

        else:
            padding = "(00)"
            currentDigits = len(res)/4

        assert currentDigits.is_integer()
        currentDigits = int(currentDigits)

        paddingLength = self.totalDigits-currentDigits
        assert paddingLength >= 0
        
        if self.reverseOrder: 
            res += padding*paddingLength
        else: 
            res =list(padding*paddingLength)+ res

        return res


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
        self.code = code
        if code == 'pos_int':
            self.subencoder = PositionalInts(params.base)
        elif code[0:3]== 'dec':
			#rn it's just for base 10..... i probably should've maade it more genreal but....  :D 
            decimalPlaces = None if code == "dec" else code[-1]
            if( decimalPlaces is not None and not decimalPlaces.isnumeric()):
                raise Exception("unexpected code, should have a zero at the end or nothing for NumberArray of decimal numbers")
            self.subencoder = PositionalRealTruncated(10, numDecimalPlaces = decimalPlaces)
        else:
            raise Exception(f"unexpected code for NumberArrayEncoder, check if theres a typo, code was {code}")
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

        if "dec" in self.code:
            m = np.zeros(tuple(shap),dtype=float)

        for val in np.nditer(m, op_flags=['readwrite']):
            v, pos = self.subencoder.parse(h)
            if v is None:
                return None
            h = h[pos:]
            val[...] = v      
        return m

class randSpacesNumberArray(NumberArray):
    def __init__(self, params, max_dim, dim_prefix, tensor_dim, code='pos_int'):
        return super().__init__(params,max_dim,dim_prefix,tensor_dim,code)
    def encode(self, vector ):
        return insertRandomSpaces(super().encode(vector), 1)
    def decode(self,lst):
        return super().decode([x for x in lst if x != " "])

    
