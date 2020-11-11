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

            # img = img.resize((46,56))


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
                print(len(point))
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

    def __setitem__(self, index, newval):
        self.classes[int(index/self.class_size)][int(index%self.class_size)] = newval


class PCA:

    def __init__(self, dataset):
        self.data = dataset


    def run(self, cut_val):

        ## array for changing cut size each fold
        cuts = []

        for fold in range(5):
            print("fold " + str(fold))
            self.train_set, self.test_set = self.data.five_fold_data(fold)

            avg, basis, points, eigs = self.train(cut_val)
            # avg, basis, points, eigs = self.train((5-fold)*cut_val)
            # print(str(len(eigs)))
            # print(",".join([str(eig) for eig in eigs]) + ",")
            acc = self.test()
            print(str(acc))


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

        # ## make each basis a unit vector, sloppy but it helps reconstruction
        basis = basis.T
        basis = numpy.array([dim/numpy.linalg.norm(dim) for dim in basis])
        basis = basis.T
        # print(basis.shape)

        # print(W)
        ## cut basis down to dimension
        basis = basis[:,:dimensions]
        eigs = W[:dimensions]

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

        return avg_vect, basis, points_for_knn, eigs


    def test(self):

        good = 0
        bad = 0
        total = len(self.test_set)
        for img, truth in self.test_set:
            # total += 1
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
                good += 1
            #     print("corr")
            # else:
            #     bad += 1
            #     print("false " + str(prediction))
        # print(str(good) + " " + str(bad) + " " + str(total))
        return good/total

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

class LDA:


    def __init__(self, dataset):
        self.data = dataset

        # self.train_set, self.test_set = self.data.five_fold_data(1)


    def run(self, dims):

        for fold in range(5):
            print("fold " + str(fold))

            self.train_set, self.test_set = self.data.five_fold_data(fold)

            basis, points, eigs = self.train(dims)
            # avg, basis, points, eigs = self.train((5-fold)*cut_val)
            # print(str(len(eigs)))
            # print(",\n".join([str(eig) for eig in eigs]) + ",")
            acc = self.test()
            print(str(acc))

    def pca_first(self, pca_dims, lda_dims):
        for fold in range(5):
            print("fold " + str(fold))

            self.train_set, self.test_set = self.data.five_fold_data(fold)

            pca = PCA(dataset)
            pca.train_set = self.train_set
            avg_vect, basis_pca, points, eig = pca.train(pca_dims)
            self.pca_avg_vect = avg_vect
            self.pca_basis = basis_pca

            print("pca done")

            ## points from pca
            for i in range(len(self.train_set)):
                self.train_set[i] = points[i][0]
            self.train_set.vector_size = len(points[0][0])

            basis, points, eigs = self.train(lda_dims)
            # avg, basis, points, eigs = self.train((5-fold)*cut_val)
            # print(str(len(eigs)))
            # print(",\n".join([str(eig) for eig in eigs]) + ",")
            acc = self.pca_first_test()
            print(str(acc))


    def train(self, dims):
        print("training")


        ## get means for each class
        means = []
        for group in self.train_set.classes:
            # print(group)
            mean = numpy.zeros(self.train_set.vector_size)
            for img in group:
                mean += img
            mean = mean/self.train_set.num_classes
            means.append(mean)
        # print("means done")

        ## within class scatter
        Sw = numpy.zeros((self.train_set.vector_size,self.train_set.vector_size))
        for i in range(len(self.train_set.classes)):
            group = self.train_set.classes[i]
            mean = means[i]
            class_scatter = numpy.zeros((self.train_set.vector_size,self.train_set.vector_size))
            # res = numpy.zeros((self.train_set.vector_size,self.train_set.vector_size))
            # print("zeros " + str(class_scatter.shape))
            for img in group:
                variance = numpy.reshape( (img-mean),(self.train_set.vector_size,1))
                class_scatter += numpy.dot(variance, variance.T)
                # print(str(variance.shape) + " : " + str(class_scatter.shape) + " = " + str(res))
                # class_scatter += res
            Sw += class_scatter

        # print("Sw scatter done " + str(Sw.shape))
        # print(Sw)

        big_mean = numpy.zeros(self.train_set.vector_size)
        for mean in means:
            big_mean += mean
        big_mean /= len(means)


        ## between class scatter
        Sb = numpy.zeros((self.train_set.vector_size,self.train_set.vector_size))
        for i in range(len(self.train_set.classes)):
            group = self.train_set.classes[i]
            mean = means[i]
            Sb += len(group)*numpy.matmul((mean-big_mean), (mean-big_mean).T)

        # print("Sb scatter done " + str(Sb.shape))

        ## solve Sb*v = lambda*Sw*v => Sw^-1*Sb*v = lambda*v
        ## eigenvalue problem
        ## psuedoinverse
        # Sw_inv = numpy.dot(numpy.linalg.inv(numpy.dot(Sw.T, Sw)), Sw.T)
        # lhs = numpy.dot(Sw_inv, Sb)
        lhs = numpy.dot(numpy.linalg.inv(Sw), Sb)
        # print("computed inverse")
        u, v = numpy.linalg.eigh(lhs)
        # print(u)
        ## we only want the c-1 basis of the largest variance so sort and slice
        combined = []
        for i in range(len(u)):
            combined.append((u[i], v[:,i]))
        combined = sorted(combined, key=lambda item: abs(item[0]), reverse=False)
        basis = numpy.array([item[1] for item in combined])
        # print([item[0] for item in combined])
        # u = [item[0] for item in combined]
        v = basis[:,:dims]
        # print(combined)
        # print(v)
        # print(basis)
        # v = v[:,:dims]


        print("Eigen decomp done " +  str(v.shape))

        points_for_knn = []
        for img, label in self.train_set:
            reduced = numpy.matmul(img, v)
            points_for_knn.append((reduced, label))

        print("points transformed")

        self.v = v
        self.points_for_knn = points_for_knn

        return v, points_for_knn, u


    def test(self):

        good = 0
        total = len(self.test_set)
        # test_set = [self.test_set[30]]
        for img, truth in self.test_set:
            # print(truth)
            ## get the reduced dimension version
            test_point = numpy.matmul(img, self.v)
            # print(test_point)

            ## find shortest distance
            min_dist = numpy.inf
            prediction = -1
            for point, label in self.points_for_knn:
                # print(point)
                dist = numpy.linalg.norm(point-test_point)
                # print(label)
                # print(dist)
                if dist < min_dist:
                    min_dist = dist
                    prediction = label
            if prediction == truth:
                good += 1
                # print("corr " + str(prediction))
            # else:
                # print("false " + str(prediction))
        # print("acc: " + str(good/total))
        return good/total

    def pca_first_test(self):

        good = 0
        bad = 0
        total = len(self.test_set)
        for img, truth in self.test_set:
            # total += 1
            # print(truth)
            ## get the reduced dimension version
            test_point = numpy.matmul(self.pca_basis.T, img-self.pca_avg_vect)
            test_point = numpy.matmul(test_point, self.v)

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
                good += 1
            #     print("corr")
            # else:
            #     bad += 1
            #     print("false " + str(prediction))
        # print(str(good) + " " + str(bad) + " " + str(total))
        return good/total


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
    dataset = DataSet(40, 10)
    # dataset.load("./smallset")
    dataset.load("./Face Data for Homework/ATT")
    print(model)
    print(cut1)


    if model == "pca":
        # run PCA
        pca = PCA(dataset)
        pca.run(cut1)

    elif model == "lda":
        ## run LDA
        lda = LDA(dataset)
        lda.run(cut1)
    elif model == "both":
        ## run combined
        lda = LDA(dataset)
        lda.pca_first(cut1, cut2)
    else:
        print("model must be [pca, lda, both]")
