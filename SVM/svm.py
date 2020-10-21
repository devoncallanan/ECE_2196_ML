#Devon Callanan
# Intro ML, University of Pittsburgh
# Homework 2: Support Vector Machine

from cvxopt import matrix, solvers, blas, printing
import numpy
import os
import sys
import math
from PIL import Image

class DataSet:
    """Hold the image data set and partition into classes"""

    def __init__(self, num_classes, class_size):
        # 40 classes of 10 images
        self.num_classes = num_classes
        self.class_size = class_size
        self.classes = [[None] *class_size for _ in range(num_classes)]
        self.classification = [-1]*num_classes
        self.total_points = 0
        self.vector_size = -1
        # for each image in the directory


    def load(self, path):
        image_files = os.listdir(path)
        for file in image_files:
            if file == "README":
                continue

            img = Image.open(os.path.join(path,file))
            self.total_points +=1
            # we need to split the filename to get the image class
            img_class = int(file[:-4].split("_")[0]) - 1
            img_num = int(file[:-4].split("_")[1]) - 1
            # read the image with pillow and create matrix of pixels

            point = numpy.array([float(x) for x in img.getdata()])
            # point = matrix([float(x) for x in img.getdata()])

            if self.vector_size == -1:
                self.vector_size = len(point)
            self.classes[img_class][img_num] = point

    def five_fold_data(self, fold_number):
        train = DataSet(self.num_classes, 8)
        test = DataSet(self.num_classes, 2)

        train.total_points = int(self.total_points*(4/5))
        test.total_points = int(self.total_points*(1/5))


        train.vector_size = self.vector_size
        test.vector_size = self.vector_size

        bound1 = (fold_number)*2
        bound2 = bound1 + 2
        for i in range(self.num_classes):
            master_group = self.classes[i]
            train.classes[i] = master_group[0:bound1] + master_group[bound2:]
            test.classes[i] = master_group[bound1:bound2]

        return train, test

    def set_class_label(self, class_num):
        self.classification = [-1]*self.num_classes
        self.classification[class_num] = 1

    def __len__(self):
        return self.total_points

    def __getitem__(self, index):
        # if self.class_size < 10:
        # print(str(int(index/self.class_size)) + "-" + str(int(index%self.class_size)))
        x = self.classes[int(index/self.class_size)][int(index%self.class_size)]
        y = self.classification[int(index/self.class_size)]
        return x,y


class DataSetFake:

    """Simple fake data set to validate SVM"""

    def __init__(self):
        self.data_vals = []
        # linear data
        # self.data_vals.append(matrix([3.5,4.25], (1,2)))
        # self.data_vals.append(matrix([4.,3.], (1,2)))
        # self.data_vals.append(matrix([4.,4.], (1,2)))
        # self.data_vals.append(matrix([4.5,1.75], (1,2)))
        # self.data_vals.append(matrix([4.9,4.5], (1,2)))
        # self.data_vals.append(matrix([5.,4.], (1,2)))
        # self.data_vals.append(matrix([5.5,2.5], (1,2)))
        # self.data_vals.append(matrix([5.5,3.5], (1,2)))
        # self.data_vals.append(matrix([4.,2.], (1,2)))
        # self.data_vals.append(matrix([2.,3.], (1,2)))
        # self.data_vals.append(matrix([.5,1.5], (1,2)))
        # self.data_vals.append(matrix([1.,2.5], (1,2)))
        # self.data_vals.append(matrix([1.25,.5], (1,2)))
        # self.data_vals.append(matrix([1.5,1.5], (1,2)))
        # self.data_vals.append(matrix([2.,2.], (1,2)))
        # self.data_vals.append(matrix([2.5,.75], (1,2)))
        # self.data_vals.append(matrix([3.,2.], (1,2)))
        # self.data_vals.append(matrix([5.,3.], (1,2)))

        self.data_cals.append(matrix([1.,2.], (1,2)))
        self.data_cals.append(matrix([4.,1.], (1,2)))
        self.data_cals.append(matrix([6.,4.5], (1,2)))
        self.data_cals.append(matrix([7.,2.], (1,2)))
        self.data_cals.append(matrix([4.,4.], (1,2)))
        self.data_cals.append(matrix([6.,3.], (1,2)))

        self.vector_size = 2

        # self.max = 18
        selt.total_points = 6

    def set_class_label(self, num):
        return None

    def __len__(self):
        return self.total_points

    def __getitem__(self, index):
        if index <= 4:
            return self.data_vals[index], 1
        else:
            return self.data_vals[index], -1


class SVMLinear:

    def __init__(self, data_path, C):
        # self.data = DataSetFake()
        # self.data = DataSet(2, 10)
        self.C = C
        self.data = DataSet(40, 10)
        self.data.load(data_path)
        self.models = [None]*len(self.data.classes)

    def run(self):

        avg_acc = 0
        for fold in range(5):
            train, test = self.data.five_fold_data(fold)
            self.train_set = train
            self.test_set = test
            # print(test.classes)
            # set only a single class label as 1 for OVA model
            for i in range(len(self.models)):
                train.set_class_label(i)
                self.models[i] = self.train(train)
                print("\r" + str(i), end = "")
                # train.set_class_label(i)
                # self.models[i] = self.train(train)

            print("\rTrained fold " + str(fold + 1) + ", testing now...")
            corr = 0
            for i in range(test.num_classes):
                # print(i)
                for j in range(test.class_size):
                    # print(j)
                    train.set_class_label(i)

                    x, y = test[i*test.class_size + j]
                    guess = self.test(x)
                    # print(x)
                    # print(guess)
                    if guess == i:
                        corr += 1
            avg_acc += (corr/test.total_points)*.2
            # print(" for fold " + str( fold) + " : " + str(corr/test.total_points))
        print("Linear " + str(avg_acc))


    def train(self, train):
        """One verses all (OVA) training pitts each class against all others"""
        #lest start simple.... single two class problem
        C = self.C
        # construct P
        P = matrix(0., (len(train), len(train)))
        # construct K
        K = matrix(0., (len(train), len(train)))
        #construct q
        q = matrix(1., (len(train),1))
        # construct G
        G = matrix(0., (len(train)*2, len(train)))
        # construct h
        h = matrix(0., (len(train)*2,1))
        # construct A
        A = matrix(0., (1, len(train)))
        # construct b
        b = matrix(0., (1, 1))
        for i in range(len(train)):
            G[i,i] = -1
            G[len(train) + i, i] = 1
            h[i,0] = 0
            h[len(train) + i, 0] = C
            xi, yi = train[i]
            A[0,i] = yi
            for j in range(len(train)):
                (xj, yj) = train[j]
                # inner_prod = blas.dot(xi.T,xj)
                inner_prod = numpy.dot(xi,xj)
                K[i,j] = inner_prod
                # inner_prod = 0
                # print(str(sum(xi)) + " " + str(sum(xj)))
                # print(inner_prod*yi*yj)
                P[i,j] = inner_prod*yi*yj

        # print("made matrices")
        # print(P)
        soln = solvers.qp(P,-q,G,h,A,b)
        a = soln['x']
        # print("solved")
        # recover w
        w = matrix(0, (1,train.vector_size))
        for i in range(len(train)):
            xi, yi = train[i]
            w = w + matrix(xi).T*yi*a[i]
        # recover b
        # b = 0
        # for i in range(len(train)):
        #     xi, yi = train[i]
        #     b += yi - w*matrix(xi)
        # b = b/len(train)

        b = 0
        singlesum = 0
        for i in range(len(train)):
            doublesum = 0.
            xi, yi = train[i]
            for j in range(len(train)):
                xj, yj = train[j]
                doublesum += a[j]*yj*(K[i,j])
            singlesum += yi - doublesum


        b = matrix((singlesum)/len(train))

        # print(a)
        # print(b)
        # print(b)
        # print("constructed w and b " + str(w) + " " + str(b) )
        return w, b, a

    def test(self, point):
        #TESTING
        # point = matrix(point)
        guess = 0
        max = -1
        for i in range(len(self.models)):

            # w, b, a = self.models[i]
            # sum = 0
            # for j in range(len(self.train_set)):
            #     xj, yj = self.train_set[j]
            #     # print(yj)
            #     sum += a[j]*yj*(numpy.dot(point, xj))
            # y_prime = sum + b[0,0]


            w, b , a= self.models[i]
            # print(str(xi) + " " + str(yi))
            y_prime = w*matrix(point) + b
            # print(str(y_prime))
            # cast to float
            y_prime = y_prime[0,0]


            if (y_prime < 0):
                continue
            elif y_prime > max:
                max = y_prime
                guess = i

        return guess
class SVMPoly:

    def __init__(self, data_path, C):
        # self.data = DataSetFake()
        # self.data = DataSet(2, 10)
        self.C = C
        self.data = DataSet(40, 10)
        self.data.load(data_path)
        self.models = [None]*len(self.data.classes)

    def run(self):

        avg_acc = 0
        for fold in range(5):
            train, test = self.data.five_fold_data(fold)
            self.train_set = train
            self.test_set = test
            # print(test.classes)
            # set only a single class label as 1 for OVA model
            for i in range(len(self.models)):
                train.set_class_label(i)
                self.models[i] = self.train(train)
                print("\r" + str(i), end = "")
                # train.set_class_label(i)
                # self.models[i] = self.train(train)

            print("\rTrained fold " + str(fold + 1) + ", testing now...")
            corr = 0
            for i in range(test.num_classes):
                # print(i)
                for j in range(test.class_size):
                    # print(j)
                    train.set_class_label(i)

                    x, y = test[i*test.class_size + j]
                    guess = self.test(x)
                    # print(x)
                    # print(guess)
                    if guess == i:
                        corr += 1
            avg_acc += (corr/test.total_points)*.2
            # print(" for fold " + str( fold) + " : " + str(corr/test.total_points))
        print("Poly " + str(avg_acc))

    def train(self, train):
        """One verses all (OVA) training pitts each class against all others"""
        #lest start simple.... single two class problem
        C = self.C
        # construct P
        P = matrix(0., (len(train), len(train)))
        # construct P
        K = matrix(0., (len(train), len(train)))
        #construct q
        q = matrix(1., (len(train),1))
        # construct G
        G = matrix(0., (len(train)*2, len(train)))
        # construct h
        h = matrix(0., (len(train)*2,1))
        # construct A
        A = matrix(0., (1, len(train)))
        # construct b
        b = matrix(0., (1, 1))
        for i in range(len(train)):
            G[i,i] = -1
            G[len(train) + i, i] = 1
            h[i,0] = 0
            h[len(train) + i, 0] = C
            xi, yi = train[i]
            A[0,i] = yi
            for j in range(len(train)):
                (xj, yj) = train[j]
                inner_prod = (numpy.dot(xi,xj) + 1)**2
                # apply kernel
                K[i,j] = inner_prod

                P[i,j] = inner_prod*yi*yj

        # print("made matrices")
        # print(P)
        soln = solvers.qp(P,-q,G,h,A,b)
        a = soln['x']
        self.a = a
        # print("solved")
        # recover w
        w = matrix(0, (1,train.vector_size))
        for i in range(len(train)):
            xi, yi = train[i]
            w = w + matrix(xi).T*yi*a[i]
        # recover b
        b = 0
        singlesum = 0
        for i in range(len(train)):
            doublesum = 0.
            xi, yi = train[i]
            for j in range(len(train)):
                xj, yj = train[j]
                doublesum += a[j]*yj*(K[i,j])
            singlesum += yi - doublesum


        b = matrix((singlesum)/len(train))
        # print(a)
        # print(b)
        # print("constructed w and b " + str(w) + " " + str(b) )
        return w, b, a

    def test(self, point):
        #TESTING

        guess = 0
        max = -1
        for i in range(len(self.models)):
            w, b, a = self.models[i]
            sum = 0
            for j in range(len(self.train_set)):
                xj, yj = self.train_set[j]
                sum += a[j]*yj*((numpy.dot(point, xj) + 1)**2)
            y_prime = sum + b[0,0]
            # print(str(y_prime))

            # w, b , a= self.models[i]
            # # print(str(xi) + " " + str(yi))
            # y_prime = w*(matrix(point) + 1)**2 + b
            # # print(str(y_prime))
            # # cast to float
            # y_prime = y_prime[0,0]


            if (y_prime < 0):
                continue
            elif y_prime > max:
                max = y_prime
                guess = i
        # print(guess)
        return guess


if __name__ == "__main__":
    # printing.options['width'] = -1
    # printing.options['dformat'] = '%.1f'
    # machine = SVMLinear("./smallset")
    C = 1

    if len(sys.argv) > 1:
        # assume it is p (preprocess)
        C = int(sys.argv[1])
    solvers.options['show_progress'] = False
    machine = SVMLinear("./Face Data for Homework/ATT", C)
    machine.run()
    # machine = SVMPoly("./smallset")
    machine = SVMPoly("./Face Data for Homework/ATT", C)
    machine.run()
