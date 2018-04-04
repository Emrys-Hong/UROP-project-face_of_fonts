from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
class Random_projection:

    def __init__(self,n=3,k=15):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n = n # n stands for number of instance we want to process
        self.X = self.mnist.train.images[:n,:]
        self.X = self.normalize_minmax(self.X) # normalize the data
        self.p = self.X.shape[1]
        self.k = k
    # we need to first normalize the mnist data
    def normalize_minmax(self, data):
        maximum = np.amax(data, axis=0)
        minimum = np.amin(data,axis=0)
        data = (data-minimum)/(maximum-minimum)
        return data



    # this function is to get the first n eigenvectors of svd as extra vectors
    def get_extra_vectors(self):
        # prevent nan exist in the data
        self.X[np.isnan(self.X)] = 0

        # this gets the max n index in the eigenvalues
        *args, self.extra_vectors = np.linalg.svd(self.X, full_matrices=True)[:self.k]
        return self.extra_vectors

    def stack_X_extra_vec(self):
        self.X_with_extra_vectors = np.vstack([self.X, extra_vectors])

    def extra_vectors_X_distance(self, extra_vectors):
        self.extra_vec_X_distance = np.dot(extra_vectors, self.X.T).T
        return self.extra_vec_X_distance

    # this is to calculate the distance between the estiamted vector V
    def V_V_distance(self):
        self.V_V_distance = np.dot(self.V, self.V.T)
        return self.V_V_distance

    # this is to calculate the real distance between the original vector X which used to compare later
    def X_X_distance(self):
        return np.dot(self.X, self.X.T)

    def RX(self):
        self.R = np.random.rand(self.p,self.k)
        self.V = np.dot(self.X, self.R)
        return self.V, self.R
    # the output shape of V is (110,15)

    def control_variant(self):
        extra_vec_X_distance_transpose = 0.5*extra_vec_X_distance.T
        length = len(extra_vec_X_distance_transpose)
        idx = list(range(length-1))
        idx.insert(0,length-1)
        for i in range(length):
            extra_vec_X_distance_transpose = extra_vec_X_distance_transpose[idx] # alternating changing the order of the extra_vec_X_distance_transpose matrix so can reduce the loop to only one loop
            extra_vector_distance_matrix = np.dot(extra_vec_X_distance, extra_vec_X_distance_transpose)
            self.V_V_distance -= extra_vector_distance_matrix



if __name__ == '__main__':
    projection1 = Random_projection() # fill in the parameter here
    extra_vectors = projection1.get_extra_vectors()
    extra_vec_X_distance = projection1.extra_vectors_X_distance(extra_vectors)
    projection1.RX()
    V_V_distance = projection1.V_V_distance()
    projection1.control_variant()
    print(projection1.V_V_distance)




# Question1: if we already calculated the dot product between x^T and x, why do we still need to estimate the relative distance ans: because it consumes less time for singular vectors decomposition
'''things to improve and fix
1. is it one and one linear combination for extra_vectors without linear combination with itself
2. why is it some value bigger than one does it supposed to be like that ?
3. how to calculate less about the svd
4. how to replace loops with matrix multiplication'''
