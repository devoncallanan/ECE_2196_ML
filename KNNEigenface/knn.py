import numpy
from PIL import Image
import os
import sys


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

            # point = numpy.array(img.getdata())
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
        y = int(index/self.class_size)
        # y = self.classification[int(index/self.class_size)]
        return x,y

class PCA:

    def __init__(self, dataset):
        self.data = dataset

        self.train_set, self.test_set = self.data.five_fold_data(1)


    def train(self, dimensions):
        """find principle components and calculate limited dim points for 1nn"""

        ## get average image vector
        sum = numpy.zeros(self.train_set.vector_size)
        for x, label in self.train_set:
            sum += x
        avg_vect = sum / self.train_set.total_points

        ## normalize all images by average and create A matrix
        A = numpy.zeros((self.train_set.vector_size,self.train_set.num_classes*self.train_set.class_size))
        for i in range(len(self.train_set)):
            x, label = self.train_set[i]
            A[:,i] = x - avg_vect

        # print(A.shape)

        # ## calculate the SVD
        # U, S, V = numpy.linalg.svd(A)
        # ## V is the new basis containing n dimensions (each dim is an eigenface)
        # basis = U[:,:dimensions]
        # print(basis.shape)

        ## trick for fast eigenvector calculation
        L = numpy.matmul(A.T, A)

        W, V = numpy.linalg.eig(L)

        basis = numpy.matmul(A, V)

        ## make each basis a unit vector, sloppy but it helps reconstruction
        basis = basis.T
        basis = numpy.array([dim/numpy.linalg.norm(dim) for dim in basis])
        basis = basis.T

        ## created reduced dim training data vectors
        points_for_knn = []
        for i in range(len(self.train_set)):
            imgx, label = self.train_set[i]
            imgx = imgx - avg_vect
            point = numpy.matmul(basis.T, imgx)
            points_for_knn.append((point, label))

        self.avg_vect = avg_vect
        self.basis = basis
        self.points_for_knn = points_for_knn

        return avg_vect, basis, points_for_knn


    def test(self):

        test_set = [self.test_set[30]]
        for img, truth in self.test_set:
            # print(truth)
            ## get the reduced dimension version
            test_point = numpy.matmul(self.basis.T, img-self.avg_vect)

            ## find shortest distance
            min_dist = numpy.inf
            prediction = -1
            for point, label in self.points_for_knn:
                dist = numpy.linalg.norm(point-test_point)
                # print(label)
                # print(dist)
                if dist < min_dist:
                    min_dist = dist
                    prediction = label
            if prediction == truth:
                print("corr")
            else:
                print("false " + str(prediction))

    def reconstruct(self, point):

        img = numpy.zeros(self.train_set.vector_size)
        print(point.shape)
        print(basis[:,0].shape)
        for i in range(len(point)):
            img += point[i]*self.basis[:,i]
        img += self.avg_vect
        self.show_img(img)



    def show_img(self, pixels):
        img_vect = numpy.reshape(pixels, (112,92))
        Image.fromarray(img_vect).show()
        input()

    def run(self):

        ## get average image vector
        sum = numpy.zeros(self.train_set.vector_size)
        for x, label in self.train_set:
            sum += x
        avg_vect = sum / self.train_set.total_points

        # img_vect = numpy.reshape(avg_vect, (112,92))
        # img = Image.fromarray(img_vect)
        # img.show()

        ## normalize all images by average and create A matrix
        A = numpy.zeros((self.train_set.num_classes*self.train_set.class_size, self.train_set.vector_size))
        for i in range(len(self.train_set)):
            x, label = self.train_set[i]
            A[i] = x - avg_vect

        print(A.shape)

        sigma = numpy.matmul(numpy.transpose(A),A)
        L = numpy.matmul(A,numpy.transpose(A))
        print(L.shape)

        # w, v = numpy.linalg.eig(L)
        # print(v)
        U, S, V = numpy.linalg.svd(A)
        print(V.shape)
        basis = V

        # basis = [numpy.matmul(eig, A) for eig in v]

        # for eig in basis:
        #     print(eig)
        #     img_vect = numpy.reshape(eig, (112,92))
        #     img = Image.fromarray(img_vect)
        #     img.show()
        #     input()

        imgx, label = self.train_set[0]
        # a = numpy.dot()
        # a = numpy.zeros((1,self.train_set.num_classes*self.train_set.class_size))
        a = numpy.zeros((1,self.train_set.vector_size))
        for i in range(100):
            dim = basis[i]
            a[0][i] = numpy.dot(numpy.transpose(dim), imgx - avg_vect)

        # reconstruct
        imgy = numpy.zeros((1, self.train_set.vector_size))
        for i in  range(100):
            dim = basis[i]
            res = a[0][i] * dim
            # print(dim)
            # print(a[0][i])
            # print(res)
            imgy += res

        imgy += avg_vect
        print(imgy)
        img_vect = numpy.reshape(imgy, (112,92))
        img = Image.fromarray(img_vect)
        img.show()



if __name__ == "__main__":
    # printing.options['width'] = -1
    # printing.options['dformat'] = '%.1f'
    # machine = SVMLinear("./smallset")
    # C = 1
    #
    # if len(sys.argv) > 1:
    #     # assume it is p (preprocess)
    #     C = int(sys.argv[1])
    dataset = DataSet(40, 10)
    # dataset.load("./smallset")
    dataset.load("./Face Data for Homework/ATT")

    pca = PCA(dataset)
    avg, basis, points = pca.train(1000)
    pca.test()

    # pca.reconstruct(points[0])
    # pca.run()
