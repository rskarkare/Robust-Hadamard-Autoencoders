import numpy as np
import numpy.linalg as nplin
import tensorflow as tf
from BasicAutoencoder import DeepAE as DAE
from shrink import l1shrink as SHR 
import PIL.Image as Image
import ImShow as I
import random
import matplotlib.pyplot as plt
import os
class RDAE(object):
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    3.0
    complete: 2/12/2018
    changes: delete unused parameter, move shrink function to other file
    update: 03/15/2019
        update to python3 
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-7):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]
        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def fit(self,X, sess, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=50, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]

        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        self.hadamard_train = np.array(DAE.new_variable)
        self.cost = list()

        mu = (X.size) / (4.0 * nplin.norm(X,1))
        print ("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S

        XFnorm = nplin.norm(X,'fro')
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            print ("mu: ", mu)
            print ("XFnorm: ", XFnorm)

        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            #self.L = X - self.S
            ## Using L to train the auto-encoder
            self.cost.append(self.AE.fit(X = X, sess = sess, S = self.S,h = self.hadamard_train,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose))
            ## get optmized L
            self.L = self.AE.getRecon(X = X, sess = sess)
            ## alternating project, now project to S
            self.S = SHR.shrink(self.lambda_/np.min([mu,np.sqrt(mu)]), (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if verbose:
                print ("c1: ", c1)
                print ("c2: ", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S
            
        return self.L , self.S, self.cost
    
    def transform(self, X, sess):
        #L = X - self.S
        return self.AE.transform(X = X, sess = sess)
    
    def getRecon(self, X, sess):
        return self.AE.getRecon(X, sess = sess)

def corrupt(X,corNum=10):
    N,p = X.shape[0],X.shape[1]
    for i in range(N):
        loclist = np.random.randint(0, p, size = corNum)
        for j in loclist:
            if X[i,j] > 0.5:
                X[i,j] = 0
            else:
                X[i,j] = 1
    return X 
    
if __name__ == "__main__":
    cor_list = [10, 20, 30, 40, 50, 60, 70, 100,150, 200, 250, 300, 350]
    lam_list = [0.00001,0.0001, 0.001,0.01, 0.1,0.5, 1.0, 5.0,10.0,50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0,3000.0]
    with tf.Session() as sess:
        for cor in cor_list:
            X_train = np.load(r"./data.npk", allow_pickle=True)[:500]
            X_test = np.load(r"./data.npk", allow_pickle=True)[500:600]
            X_test.dump(r'./X_test.npk')
            Xclean = X_train.copy()
            inputsize = (28,28)
            image_clean = Image.fromarray(I.tile_raster_images(X=Xclean,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
            image_clean.save(r"Xclean.pdf")
            X_train = corrupt(X_train, corNum=cor)
            X_test = corrupt(X_test, corNum=cor)
           # DAE.new_variable = DAE.new_variable.reshape(-1, 784)
            X_train = np.where(DAE.new_variable==0.0,0.5,X_train)
            DAE.new_variable_test = DAE.new_variable_test.reshape(-1, 784)
            X_test = np.where(DAE.new_variable_test==0.0,0.5,X_test)  	
            up_folder = r"Cor"+str(cor)
            if not os.path.isdir(up_folder):
                os.makedirs(up_folder)
            os.chdir(up_folder)
            for lamda in lam_list:
                folder = r"lam" + str(lamda)
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                os.chdir(folder)
                rae = RDAE(sess = sess, lambda_= lamda, layers_sizes=[784,400])
                inputsize = (28,28)
                L, S, cost = rae.fit(X_train ,sess = sess, learning_rate=0.01, batch_size = 100
                , inner_iteration = 50, iteration=5, verbose=True)
                cost = np.array(cost).flatten()
                plt.plot(np.arange(len(cost)),cost, '-', label='Lambda: '+str(lamda), linewidth=2)
                plt.xlabel('Epochs',fontsize=20)
                plt.ylabel('Cost',fontsize=20)
                plt.grid()
                L.dump(r"Lowrank.npk")
                image_S = Image.fromarray(I.tile_raster_images(X=X_train,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
                image_S.save(r"X_train.pdf")
                image_S = Image.fromarray(I.tile_raster_images(X=S,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
                image_S.save(r"S.pdf")
                rH = rae.transform(X_test,sess=sess)
                rH.dump(r"rh_test.npk")
                R = rae.getRecon(X_test, sess=sess)
                R.dump(r"rR_test.npk")
                image_R = Image.fromarray(I.tile_raster_images(X=L,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
                image_R.save(r"Lowrank.pdf")
                image_R = Image.fromarray(I.tile_raster_images(X=R,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
                image_R.save(r"Recon.pdf")
                os.chdir('../')
            plt.legend(loc='upper right')
            plt.savefig('./cost_function2.pdf')
            os.chdir('../')


