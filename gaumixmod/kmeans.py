from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


class Kmeans(object):
    """
    K means method.
    Too lazy to brush it...
    """

    def __init__(self, data, means):

        self.data  = data
        self.means = means

        assert(data.shape[1] == means.shape[1]), "Dimension should be consistent!"

        self.nn, self.mm = self.data.shape

        self.kk = self.means.shape[0]

        self.nchg = 1

        self.assign = np.zeros(self.nn, dtype = 'int64')
        self.count  = np.zeros(self.kk, dtype = 'int64')


        while(self.nchg != 0):

            self._estep()
            self._mstep()



    def _estep(self):


        self.nchg = 0
        self.count = np.zeros(self.nn, dtype = 'int64')

        for n in range(self.nn):

            dmin = 9.99e99

            for k in range(self.kk):
                
                d = self.data[n] - self.means[k]
                d = np.sqrt(d.dot(d))

                if d < dmin:

                    dmin = d
                    kmin = k

            if kmin != self.assign[n]:

                self.nchg += 1

            self.assign[n] = kmin
            self.count[kmin] += 1


        return self.nchg


    def _mstep(self):

        self.means = np.zeros([self.kk, self.mm])

        for n in range(self.nn):

            self.means[self.assign[n]] += self.data[n]

        for k in range(self.kk):

            self.means[k] /= self.count[k]

            
                

if __name__ == "__main__":

    
    features, target = make_blobs(n_samples = 1000, n_features = 2, centers = 3, cluster_std = 0.5, shuffle = True, random_state = 1)

    #plt.scatter(features[:, 0], features[:, 1], c = target)

    #plt.show()

    centers = np.array([[2.0, 2.0],
                        [-1.0, -5.0],
                        [-5.0, -1.0]])

    
    model = Kmeans(data = features, means = centers)


    plt.scatter(features[:, 0], features[:, 1], c = target)
    plt.show()

    plt.scatter(features[:, 0], features[:, 1], c = model.assign)
     
    plt.show()
        

        

        

