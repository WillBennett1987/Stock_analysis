import numpy as np
import pandas as pd
import math

def round_seq(seq, acc):
    return [round(x, acc) for x in seq]
    
class Seq(object):  # make a new instance of a sequence object
    def __init__(self, x, y):  # old : the parameter n1 is the first row in the sequence and so for the Gann sequence it would look like [1, 2, 3, 2/3] 0:index, 1: the x parameter in the sequece, 2: the y parameter in the sequence, 3: the first ratio (x/y) or ([1]/[2]) could do this for simplicity
        """
        x -
        dtype : int
        it is the parameter in the sequence which with y is able to create the sequence
        
        y - 
        dtype : int
        it is the parameter in the sequence which with x is able to create the sequence

        """
        self.start_x = x
        self.start_y = y
        self.n1 = [1, x, y, x/y]
        self.myArray = [self.n1]

    def testRow(self, x, y, n):  # tests the method to see if it is right
        rem = n % 2
        # print(rem) # only for debugging
        if rem == 1:
            z = x / y
            if z < 1.0 and z > 0.5:  # need to check with dad and see if its equal to 1 or 0.5 is ok
                return [True, z] 
            else:
                return [False, 0]

        if rem == 0:
            z = y / x
            if z < 1.0 and z > 0.5:  # need to check with dad and see if its equal to 1 or 0.5 is ok
                return [True, z]
            else:
                return [False, 0]
        else:
            return [False, 0]

    def makeRow(self,x2,y2,n):#
        condis = self.testRow(x2, y2, n)
        # print(type(condis))
        # print(condis[0],condis[1])
        if condis[0]:
            z2 = condis[1]
            return [n, x2, y2, z2]  # give the values back and ends the function
        else:
            return False

    def addElem(self, xn):
        n = xn[0]
        n = n + 1
        x1 = xn[1]  # gets number out of array
        y1 = xn[2]
        z1 = xn[3]

        x2 = x1  # tries the first method
        y2 = 3 * y1
        test = self.makeRow(x2,y2,n)
        if test != False:
            return test


        x2 = 2 * x1  # tries the second method
        y2 = y1
        test = self.makeRow(x2, y2, n)
        if test != False:
            return test

        # if that didnt work then we run the third method
        x2 = 2 * x1
        y2 = 3 * y1
        test = self.makeRow(x2, y2, n)
        if test != False:
            return test
        else:
            z2 = "not found"
            print(f"Error both methods are incorrect with n: {n} x1: {x1},y1: {y1}, z1: {z1} to \n n{n + 1}, x2: {x2}, y2: {y2}, z2: {z2}")

    def run(self, length):
        #    print("   n   |   x   |   y   |   z  ")
        for i in range(length):
            MAX = len(self.myArray) - 1
            elem = self.addElem(self.myArray[MAX])
            str(elem[0])
            str(elem[1])
            str(elem[2])
            str(elem[3])
            self.myArray.append(elem)


