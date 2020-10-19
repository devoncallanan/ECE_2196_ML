#Devon Callanan
# Intro ML, University of Pittsburgh
# Homework 2: Support Vector Machine

from cvxopt import matrix, solvers, blas
import os
import sys
import math
from PIL import Image

class DataSet:
    """Hold the image data set and partition into classes"""

    def __init__(self, path):
        # 40 classes of 10 images
        self.classes = [[None]*10]*40
        self.classification = [-1]*40
        self.max = 0
        # for each image in the directory
        image_files = os.listdir(path)
        for file in image_files:
            if file == "README":
                continue
            img = Image.open(os.path.join(path,file))
            self.max +=1
            # we need to split the filename to get the image class
            img_class = int(file[:-4].split("_")[0]) - 1
            img_num = int(file[:-4].split("_")[1]) - 1
            # read the image with pillow and create matrix of pixels
            self.classes[img_class][img_num] = matrix([float(x) for x in img.getdata()])

    def set_class_label(self, class_num):
        self.classification = [-1]*40
        self.classification[class_num] = 1

    def __len__(self):
        return self.max

    def __getitem__(self, index):
        x = self.classes[int(index/10)][int(index%10)]
        y = self.classification[int(index/10)]
        return x,y
    #
    # def __iter__(self):
    #     return self
    #
    # def __next__(self):
    #     self.current += 1
    #     if self.current == self.max:
    #         raise StopIteration
    #     else:
    #         x = self.classes[self.current/10][self.current%10]
    #         y = self.classification[self.current/10]
    #         return (x,y)


class SVM:

    def __init__(self, data_path):
        self.data = DataSet(data_path)

    def train(self):
        """One verses all (OVA) training pitts each class against all others"""
        #lest start simple.... single two class problem
        self.data.set_class_label(0)
        C = 15
        # construct P
        P = matrix(0., (len(self.data), len(self.data)))
        #construct q
        q = matrix(1., (len(self.data),1))
        # construct G
        G = matrix(0., (len(self.data)*2, len(self.data)))
        # construct h
        h = matrix(0., (len(self.data)*2,1))
        # construct A
        A = matrix(0., (1, len(self.data)))
        # construct b
        b = matrix(0., (1, 1))
        for i in range(len(self.data)):
            G[i,i] = -1
            G[len(self.data) + i, i] = 1
            h[i,0] = 0
            h[len(self.data) + 1, 0] = C
            xi, yi = self.data[i]
            A[0,1] = yi
            for j in range(len(self.data)):
                (xj, yj) = self.data[j]
                inner_prod = blas.dot(xi,xj)
                # print(str(sum(xi)) + " " + str(sum(xj)))
                # print(inner_prod*yi*yj)
                P[i,j] = inner_prod*yi*yj

        print(P)
        soln = solvers.qp(P,q,G,h,A,b)
        print(soln['x'])

        # recover w

        # recover b



if __name__ == "__main__":
    machine = SVM("./smallset")
    # machine = SVM("./Face Data for Homework/ATT")
    machine.train()
