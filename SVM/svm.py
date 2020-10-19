#Devon Callanan
# Intro ML, University of Pittsburgh
# Homework 2: Support Vector Machine

from cvxopt import matrix, solvers, blas, printing
import os
import sys
import math
from PIL import Image

class DataSet:
    """Hold the image data set and partition into classes"""

    def __init__(self, path):
        # 40 classes of 10 images
        self.classes = [[None] *10 for _ in range(40)]
        self.images = [[None] *10 for _ in range(40)]
        self.classification = [-1]*40
        self.max = 0
        self.vector_size = -1
        # for each image in the directory
        image_files = os.listdir(path)
        for file in image_files:
            if file == "README":
                continue
            # print(file)
            img = Image.open(os.path.join(path,file))
            self.max +=1
            # we need to split the filename to get the image class
            img_class = int(file[:-4].split("_")[0]) - 1
            img_num = int(file[:-4].split("_")[1]) - 1
            # read the image with pillow and create matrix of pixels
            # print(file + " " + str(img_class) + ":" + str(img_num))

            point = matrix([float(x) for x in img.getdata()])
            # print(point.T)
            if self.vector_size == -1:
                self.vector_size = len(point)
            self.classes[img_class][img_num] = point.T
            self.images[img_class][img_num] = img

    def set_class_label(self, class_num):
        self.classification = [-1]*40
        self.classification[class_num] = 1

    def __len__(self):
        return self.max

    def __getitem__(self, index):
        # print(str(int(index/10)) + "-" + str(int(index%10)))
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

class DataSetFake:

    def __init__(self):
        self.data_vals = []
        self.data_vals.append(matrix([3.5,4.25], (1,2)))
        self.data_vals.append(matrix([4.,3.], (1,2)))
        self.data_vals.append(matrix([4.,4.], (1,2)))
        self.data_vals.append(matrix([4.5,1.75], (1,2)))
        self.data_vals.append(matrix([4.9,4.5], (1,2)))
        self.data_vals.append(matrix([5.,4.], (1,2)))
        self.data_vals.append(matrix([5.5,2.5], (1,2)))
        self.data_vals.append(matrix([5.5,3.5], (1,2)))
        self.data_vals.append(matrix([4.,2.], (1,2)))
        self.data_vals.append(matrix([2.,3.], (1,2)))
        self.data_vals.append(matrix([.5,1.5], (1,2)))
        self.data_vals.append(matrix([1.,2.5], (1,2)))
        self.data_vals.append(matrix([1.25,.5], (1,2)))
        self.data_vals.append(matrix([1.5,1.5], (1,2)))
        self.data_vals.append(matrix([2.,2.], (1,2)))
        self.data_vals.append(matrix([2.5,.75], (1,2)))
        self.data_vals.append(matrix([3.,2.], (1,2)))
        self.data_vals.append(matrix([5.,3.], (1,2)))

        self.vector_size = 2

        self.max = 18

    def set_class_label(self, num):
        return None

    def __len__(self):
        return self.max

    def __getitem__(self, index):
        if index <= 9:
            return self.data_vals[index], 1
        else:
            return self.data_vals[index], -1


class SVM:

    def __init__(self, data_path):
        # self.data = DataSetFake()
        self.data = DataSet(data_path)
        self.data.set_class_label(0)
        #
        # for i in range(len(self.data)):
        #     x, y = self.data[i]
        #     self.data.images[int(i/10)][int(i%10)].show()
        #     stop = input()

    def train(self):
        """One verses all (OVA) training pitts each class against all others"""
        #lest start simple.... single two class problem
        self.data.set_class_label(0)
        C = 10
        # construct P
        P = matrix(0., (len(self.data), len(self.data)))
        #construct q
        q = matrix(1., (len(self.data),1))
        # construct G
        G = matrix(0., (len(self.data)*2, len(self.data)))
        # construct h
        h = matrix(0., (len(self.data)*2,1))
        # # construct G
        # G = matrix(0., (len(self.data), len(self.data)))
        # # construct h
        # h = matrix(0., (len(self.data),1))
        # construct A
        A = matrix(0., (1, len(self.data)))
        # construct b
        b = matrix(0., (1, 1))
        for i in range(len(self.data)):
            G[i,i] = -1
            G[len(self.data) + i, i] = 1
            h[i,0] = 0
            h[len(self.data) + i, 0] = C
            xi, yi = self.data[i]
            A[0,i] = yi
            for j in range(len(self.data)):
                (xj, yj) = self.data[j]
                inner_prod = blas.dot(xi.T,xj)
                # print(str(sum(xi)) + " " + str(sum(xj)))
                # print(inner_prod*yi*yj)
                P[i,j] = inner_prod*yi*yj

        # P = -P
        # print(P)
        # print(q)
        # print(G)
        # print(h)
        # print(A)
        # print(b)
        soln = solvers.qp(P,-q,G,h,A,b)
        a = soln['x']

        # recover w
        w = matrix(0, (1,self.data.vector_size))
        for i in range(len(self.data)):
            xi, yi = self.data[i]
            w = w + xi*yi*a[i]
        # w =
        # recover b
        b = 0
        for i in range(len(self.data)):
            xi, yi = self.data[i]
            b += yi - w*xi.T
        b = b/len(self.data)
        print(a)
        print(w)
        print(b)

        #TESTING
        for i in range(len(self.data)):
            xi, yi = self.data[i]
            # print(str(xi) + " " + str(yi))
            y_prime = w*xi.T + b
            # print(str(y_prime) + " " + str(yi))
            if (yi < 0 and y_prime[0,0] < 0) or (yi > 0 and y_prime[0,0] > 0):
                print("COR")
            else:
                print("False")


if __name__ == "__main__":
    # printing.options['width'] = -1
    # printing.options['dformat'] = '%.1f'
    machine = SVM("./smallset")
    # machine = SVM("./Face Data for Homework/ATT")
    machine.train()
