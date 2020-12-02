import numpy
from scipy.sparse import csr_matrix
from PIL import Image
from multiprocessing import Pool
import random
import os
import sys
import math



class DataSet:
    """Hold the image data set and partition into classes
        each data point is a tuple of the data and the class
    """

    def __init__(self, num_classes, class_size):
        # 40 classes of 10 images
        self.num_classes = num_classes
        self.class_size = class_size
        self.classes = [[None] *class_size for _ in range(num_classes)]
        # self.data = []
        # self.classification = [-1]*num_classes
        self.total_points = 0
        self.vector_size = 120*90
        # for each image in the directory


    def load(self, path):
        samples = os.listdir(path)
        for sample in samples:
            # print(sample)
            if sample == "all.txt~":
                continue
            sample_dir = os.path.join(path,sample)
            img_files = os.listdir(sample_dir)
            # we need to split the filename to get the image class
            img_class = int(sample[-3:]) - 1
            img_num = 0
            pool = Pool(processes=6)
            files = [os.path.join(sample_dir,img_file) for img_file in img_files]
            self.classes[img_class] = [(point, img_class) for point in pool.map(DataSet.parallel_load, files)]


    def parallel_load(file):
        img = Image.open(file)

        img = img.resize((120,90))

        point = numpy.array([ 1. if x[0] < 128 else 0. for x in img.getdata()]).reshape((120*90,1))
        return point



    def split(self):
        train = DataSet(self.num_classes, 50)
        test = DataSet(self.num_classes, 5)

        train.total_points = self.num_classes*50
        test.total_points = self.num_classes*5


        train.vector_size = self.vector_size
        test.vector_size = self.vector_size




        for i in range(self.num_classes):
            master_group = self.classes[i]
            random.shuffle(master_group)
            train.classes[i] = master_group[0:50]
            test.classes[i] = master_group[50:]

        return train, test

    def __len__(self):
        return self.total_points

    def __getitem__(self, index):

        return self.classes[int(index/self.class_size)][int(index%self.class_size)]


    def __setitem__(self, index, newval):
        self.classes[int(index/self.class_size)][int(index%self.class_size)] = newval


class ANN:

    def __init__(self, dataset):

        self.dataset = dataset
        #  layer lengths
        self.l1_n = 90*120
        self.l2_n = 100
        self.l3_n = 62

        # layer weight matrices
        self.l2_w = numpy.random.randn(self.l1_n,self.l2_n)
        self.l3_w = numpy.random.randn(self.l2_n,self.l3_n)

        # bias matrices
        self.l2_b = numpy.random.randn(self.l2_n, 1)
        self.l3_b = numpy.random.randn(self.l3_n, 1)

        print("~~~~training~~~~")
        self.train()
        print("~~~~testing~~~~~")
        self.test()
        print("~~~~done~~~~~~~~")

    def train(self):

        trainset, testset = self.dataset.split()

        # point, label = testset[5]
        # img_vect = numpy.reshape(point, (90,120))
        # Image.fromarray(img_vect).show()

        # quit()

        #feed forward

        errs = []
        for _ in range(40):
            print(str(_) + "\r", end="")
            batch_it = 0
            batch_size = 1
            b2 = 0
            b3 = 0
            w2 = 0
            w3 = 0
            error = 0
            points = []
            for arr in trainset.classes[:self.l3_n]:
                points += arr
            random.shuffle(points)
            for point, label in points:
                batch_it += 1

                # Feedforeward
                l2_z = self.l2_w.T.dot(point) + self.l2_b
                l2_a = self.sigmoid(l2_z)

                l3_z = self.l3_w.T.dot(l2_a) + self.l3_b
                l3_a = self.sigmoid(l3_z)


                ## backprop
                cost_deriv = self.error_deriv(l3_a, label)
                l3_grad = cost_deriv
                l2_grad = self.l3_w.dot(l3_grad)*self.sig_deriv(l2_z)


                ## online learning
                # decay = 2 - math.log((1.1+_)/2.4, 10)
                # print(decay)
                # decay = 1
                alpha = 1
                # alpha = .7
                self.l2_b -= alpha*l2_grad
                self.l2_w -= alpha*(point.dot(l2_grad.T))
                alpha = decay
                # alpha = .7
                self.l3_b -= alpha*l3_grad
                self.l3_w -= alpha*(l2_a.dot(l3_grad.T))

                # error += self.cross_loss(l3_a, label)

            # print(str(error[0]/len(points)) + ",", end="")
            # errs.append(error[0]/len(points))
            # if _%5 == 0:
            #     print(_, end=",")
            #     self.test()
        # print(" errors ")
        # print(",".join([str(err) for err in errs]))

    def test(self):
        #feed forward
        trainset, testset = self.dataset.split()
        correct = 0
        total = 0
        points = []
        for arr in testset.classes[:self.l3_n]:
            points += arr
        for point, label in points:
            total += 1

            l2_a = self.sigmoid(self.l2_w.T.dot(point) - self.l2_b)
            # print("layer 2 " + str(l2_a))
            l3_a = self.sigmoid(self.l3_w.T.dot(l2_a) - self.l3_b)

            # print(str(label) + " " + str(l3_a.T))

            guess = 0
            for i in range(len(l3_a)):
                if l3_a[i] > l3_a[guess]:
                    guess = i
            if guess == label:
                correct += 1
            ## for printing incorrect guess images
            # else:
            #     img_vect = numpy.reshape([255. if x[0] < 1 else 0. for x in point], (90,120))
            #     Image.fromarray(img_vect).show()
            #     print(guess)
            #     input()
        # print("ACCURACY")
        print(float(correct)/total)


    def cross_loss(self, a, label):
        error = 0
        for i in range(len(a)):
            t = 1 if i == label else 0
            # print(a[i])
            e = t*numpy.log(a[i]) + (1-t)*numpy.log(1-a[i])
            # print(e)
            error -= e
        return error

    def error_deriv(self, a, label):
        t_mat = numpy.zeros((self.l3_n,1))
        t_mat[label] = 1.
        # print(str(label) + " " + str(a.T))
        del_err = (a-t_mat)*a*(1-a)
        # return a-t_mat
        return del_err

    def sig_deriv(self, z):
        # print(z)
        # return (numpy.exp(-z/100))/(1 + numpy.exp(-z/100))**2
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def sigmoid(self, z):
        # print(z)
        # print((1+numpy.exp(-z/100)))
        return 1./(1+numpy.exp(-z/10))



if __name__ == "__main__":

    ## defaults for input vals
    model = "pca"
    cut1 = 130
    cut2 = 90

    if len(sys.argv) > 1:
        # assume it is p (preprocess)
        model = sys.argv[1]
    if len(sys.argv) > 2:
        # assume it is p (preprocess)
        cut1 = int(sys.argv[2])
    if len(sys.argv) > 3:
        # assume it is p (preprocess)
        cut2 = int(sys.argv[3])
    dataset = DataSet(62, 55)
    # dataset.load("./smallset")
    dataset.load("./English/Hnd/Img")

    ann = ANN(dataset)
