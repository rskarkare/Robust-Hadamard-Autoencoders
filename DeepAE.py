import tensorflow as tf
import numpy as np
import random
X_train = np.load(r"./data.npk", allow_pickle = True)[:500]
hadamard_train = np.ones(X_train.shape)
#hadamard_train = np.ones((500, 784))
hadamard_train = hadamard_train.reshape(-1, 28, 28, 1)
block_size = 10
for i in range(len(hadamard_train)):
    x = random.randint(0,28-block_size)
    y = random.randint(0,28-block_size)
    for j in range(block_size):
        for k in range(block_size):
            hadamard_train[i][x+j][y+k] = 0.0

new_variable = hadamard_train.copy()
hadamard_train = hadamard_train.reshape(-1, 784)
X_test = np.load(r"./data.npk", allow_pickle = True)[500:600]
hadamard_test = np.ones(X_test.shape)
#hadamard_train = np.ones((500, 784))
hadamard_test = hadamard_test.reshape(-1, 28, 28, 1)
block_size = 10
for i in range(len(hadamard_test)):
    x = random.randint(0,28-block_size)
    y = random.randint(0,28-block_size)
    for j in range(block_size):
        for k in range(block_size):
            hadamard_test[i][x+j][y+k] = 0.0

new_variable_test = hadamard_test.copy()
hadamard_test = hadamard_test.reshape(-1, 784)
def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i,min(l,i+n))
class Deep_Autoencoder(object):
    def __init__(self, sess, input_dim_list=[784,400]):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        self.cost_final = []
        ## Encoders parameters
        for i in range(len(input_dim_list)-1):
            init_max_value = np.sqrt(6. / (self.dim_list[i] + self.dim_list[i+1]))
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i],self.dim_list[i+1]],
                                                             np.negative(init_max_value),init_max_value)))
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]],-0.1,0.1)))
        ## Decoders parameters
        for i in range(len(input_dim_list)-2,-1,-1):
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]],-0.1,0.1)))
        ## Placeholder for input
        self.input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        self.S = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        self.hadamard_train = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        ## coding graph :
        last_layer = self.input_x
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        self.hidden = hidden 
        ## decode graph:
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer
        #self.recon = self.recon*(1.-hadamard_train) + self.input_x*(hadamard_train)
        self.cost = 200 * tf.reduce_mean(tf.square((self.input_x*self.hadamard_train) - ((self.recon-self.S)*self.hadamard_train)))
        
        self.train_step = tf.train.AdamOptimizer().minimize(self.cost)
        sess.run(tf.global_variables_initializer())

    def fit(self, X,S,h, sess, learning_rate=0.15,
            iteration=200, batch_size=50, init=False,verbose=False):
        assert X.shape[1] == self.dim_list[0]
        if init:
            sess.run(tf.global_variables_initializer())
        sample_size = X.shape[0]
        for i in range(iteration):
            for one_batch in batches(sample_size, batch_size):
                sess.run(self.train_step,feed_dict = {self.input_x:X[one_batch], self.S:S[one_batch], self.hadamard_train:h[one_batch]})

            if verbose and i%20==0:
                e = self.cost.eval(session = sess,feed_dict = {self.input_x: X, self.S: S, self.hadamard_train: h})
                print ("    iteration : ", i ,", cost : ", e)
                self.cost_final.append(e)
        return self.cost_final

    def transform(self, X, sess):
        return self.hidden.eval(session = sess, feed_dict={self.input_x: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session = sess,feed_dict={self.input_x: X})
    
##################### test a machine with different data size#####################  
def test():
    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:1000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 1000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:10000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 10,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:20000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 20,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:50000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 50,000 Runing time:" + str(time.time() - start_time) + " s")
if __name__ == "__main__":
    import time
    import os
    os.chdir("../../")
    x = np.load(r"./data/data.npk")
    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,225,100])
        error = ae.fit(x ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 500, verbose=True)
        R = ae.getRecon(x, sess = sess)
        print ("size 100 Runing time:" + str(time.time() - start_time) + " s")
        error = ae.fit(R ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 500, verbose=True)
    #test()
