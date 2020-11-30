import numpy
from PIL import Image
from multiprocessing import Pool
import random
import os
import sys



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
            # print(self.classes[img_class])
            # for img_file in img_files:
            #     img = Image.open(os.path.join(sample_dir,img_file))
            #
            #     img = img.resize((120,90))
            #
            #
            #     # read the image with pillow and create matrix of pixels
            #     # point = numpy.array(img.getdata())
            #     # print(list(img.getdata()))
            #     # point= img.getdata()
            #     point = numpy.array([float(x[0]) for x in img.getdata()])
            #     # print(point)
            #     if self.vector_size == -1:
            #         self.vector_size = 2
            #         # self.vector_size = len(point)
            #     self.classes[img_class][img_num] = point, img_class
            #     img_num += 1
            #     self.total_points += 1
            #     img.close()

    def parallel_load(file):
        img = Image.open(file)

        img = img.resize((120,90))

        point = numpy.array([float(x[0]/255) for x in img.getdata()]).reshape((120*90,1))
        return point



    def split(self):
        train = DataSet(self.num_classes, 50)
        test = DataSet(self.num_classes, 5)

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
        self.l3_n = 10

        # layer weight matrices
        self.l2_w = numpy.random.randn(self.l1_n,self.l2_n)
        # print(self.l1_w)
        self.l3_w = numpy.random.randn(self.l2_n,self.l3_n)
        # print(self.l2_w)

        # bias matrices
        self.l1_b = numpy.random.randn(self.l1_n, 1)
        self.l2_b = numpy.random.randn(self.l2_n, 1)
        self.l3_b = numpy.random.randn(self.l3_n, 1)

        self.train()
        self.test()

    def train(self):

        #feed forward
        for _ in range(100):
            print(_)
            b2 = 0
            b3 = 0
            w2 = 0
            w3 = 0
            points = []
            for arr in dataset.classes[:10]:
                points += arr
            random.shuffle(points)
            for point, label in points:
            # for point, label in self.dataset:
                # print(label)
                l2_z = self.l2_w.T.dot(point) + self.l2_b
                l2_a = self.sigmoid(l2_z)
                # print(l2_a.shape)
                # print(l2_a)
                # print("layer 2 " + str(l2_a.shape))
                l3_z = self.l3_w.T.dot(l2_a) + self.l3_b
                l3_a = self.sigmoid(l3_z)
                # print(l3_a.shape)
                # print(l3_a.T)
                # print("layer 3 " + str(l3_a.shape))
                cost_deriv = self.error_deriv(l3_a, label)
                l3_loss_deriv = cost_deriv
                # print(cost_deriv.shape)
                # print(cost_deriv)
                # l3_loss_deriv = cost_deriv*self.sig_deriv(l3_z)
                l2_loss_deriv = self.l3_w.dot(l3_loss_deriv)*self.sig_deriv(l2_z)

                # print("l3 and l2 loss derivs ")
                # print(l3_loss_deriv)
                # print(l2_loss_deriv)

                # update weights
                # alpha = .1
                # b2 += alpha*l2_loss_deriv
                #
                # b3 += alpha*l3_loss_deriv
                #
                # w2 += alpha*(point.dot(l2_loss_deriv.T))
                # w3 += alpha*(l2_a.dot(l3_loss_deriv.T))

                ## online learning
                alpha = .1
                self.l2_b -= alpha*l2_loss_deriv
                self.l2_w -= alpha*(point.dot(l2_loss_deriv.T))
                alpha = .005
                self.l3_b -= alpha*l3_loss_deriv
                self.l3_w -= alpha*(l2_a.dot(l3_loss_deriv.T))



                # error = self.cross_loss(l3_a, label)
            # self.l2_b -= b2
            # self.l3_b -= b3
            # self.l2_w -= w2
            # self.l3_w -= w3
            # print(error)

    def test(self):
        #feed forward
        correct = 0
        total = 0
        points = []
        for arr in dataset.classes[:10]:
            points += arr
        for point, label in points:
            total += 1
            # print(point)
            # print(self.l2_w)
            # print(self.l3_w)
            # print(self.l2_b)
            # print(point)
            l2_a = self.sigmoid(self.l2_w.T.dot(point) - self.l2_b)
            # print("layer 2 " + str(l2_a))
            l3_a = self.sigmoid(self.l3_w.T.dot(l2_a) - self.l3_b)

            print(str(label) + " " + str(l3_a.T))

            guess = 0
            for i in range(len(l3_a)):
                if l3_a[i] > l3_a[guess]:
                    guess = i
            if guess == label:
                correct += 1
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
        t_mat = numpy.zeros((10,1))
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
    # point, label = dataset[2000]
    # print(label)
    # img_vect = numpy.reshape(point, (90,120))
    # Image.fromarray(img_vect).show()

    ann = ANN(dataset)
