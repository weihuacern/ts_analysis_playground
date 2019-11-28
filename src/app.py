import pywt
import numpy as np
import tensorflow as tf

#from tensorflow.contrib import rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def entry():
    X_fill = load_data("train_filled.csv")
    X_wv = denoise(X_fill)
    X_train, Y_train, X_test, Y_test = split(X_wv)
    Y_sae, Y_sae_test = stackedAutoencoders(X_train, X_test)

    Y_hat, Y_hat_train = LSTM(Y_sae, Y_train, Y_sae_test)
    accuracy_test = metric(Y_hat, Y_test)
    accuravy_train = metric(Y_hat_train, Y_train)

    print("Training Set Accuracy: " + str(accuravy_train*100) + "%")
    print("Test Set Accuracy: " + str(accuracy_test*100) + "%")

#loads data
def load_data(filename):
    return np.loadtxt(filename, delimiter=',')

#applies wavelet transform
def denoise(X):
    m, n = X.shape

    first_part = np.zeros((m, 28))
    third_part = np.zeros((m, 64))
    for row in range(m):
        for col1 in range(28):
            first_part[row][col1] = X[row][col1]
        for col2 in range(64):
            third_part[row][col2] = X[row][col2]

    wav = pywt.Wavelet('haar')

    D = np.zeros((m, 120))
    for i, xi in enumerate(X):
        coeffs = pywt.wavedec(xi[28:147], wav, mode='symmetric', level=1)
        cA, cD = coeffs
        cA = np.array(cA)
        cD = np.array(cD)
        D[i][:] = np.concatenate((cA, cD))
    return np.concatenate((first_part, D, third_part), axis=1)

#splits data into X train, Y train, X test, Y test
def split(X_raw):
    m, n = X_raw.shape
    np.random.shuffle(X_raw)
    X_train = np.zeros((30000, 147))
    Y_train = np.zeros((30000, 62))
    X_test = np.zeros((10000, 147))
    Y_test = np.zeros((10000, 62))
    for row in range(m):
        if row < 30000:
            for col1 in range(1, 148):
                X_train[row][col1-1] = X_raw[row][col1]
            for col2 in range(148, 210):
                Y_train[row][col2-148] = X_raw[row][col2]
        else:
            for col1 in range(1, 148):
                X_test[row-30000][col1-1] = X_raw[row][col1]
            for col2 in range(148, 210):
                Y_test[row-30000][col2-148] = X_raw[row][col2]
    return X_train.T, Y_train.T, X_test.T, Y_test.T

# Trains the stacked Autoencoders and then passes both X_train and X_test
# into the SAE for next steps. 147->74->50->74->147
def stackedAutoencoders(X_input_train, X_input_test):
    # Define parameters
    num_examples = 30000
    num_inputs = 147
    num_hid1 = 74
    num_hid2 = 50
    num_hid3 = num_hid1
    num_output = num_inputs
    lr = 0.01
    actf = tf.nn.relu
    num_epoch = 1
    batch_size = 200

    # Create inputs
    X = tf.placeholder(tf.float32, shape=[num_inputs, 30000])
    X_test = tf.placeholder(tf.float32, shape=[num_inputs, 10000])

    # Define variables
    W1 = tf.get_variable("W1", [74, 147], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [74, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [50, 74], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [50, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [74, 50], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [74, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [147, 74], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [147, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}

    hid_layer1_train = actf(tf.matmul(W1, X)+b1)
    hid_layer2_train = actf(tf.matmul(W2, hid_layer1_train)+b2)
    hid_layer3_train = actf(tf.matmul(W3, hid_layer2_train)+b3)
    output_layer = actf(tf.matmul(W4, hid_layer3_train)+b4)

    hid_layer1_test = actf(tf.matmul(W1, X_test)+b1)
    hid_layer2_test = actf(tf.matmul(W2, hid_layer1_test)+b2)

    loss = tf.reduce_mean(tf.square(output_layer-X))

    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(train, feed_dict={X:X_input_train})

        y_sae_train = sess.run(hid_layer2_train, feed_dict={X:X_input_train})
        y_sae_test = sess.run(hid_layer2_test, feed_dict={X_test:X_input_test})

        return y_sae_train, y_sae_test

# Creating LSTM
def myLSTM(X, Y, X_test):
    #Dropout parameter
    drop = 0.1

    # Initialising the RNN
    regressor = Sequential()

    # Adding some Dropout regularisation and more RNN layers
    regressor.add(Dropout(drop))
    regressor.add(Sequential())
    regressor.add(Dropout(drop))
    regressor.add(Sequential())
    regressor.add(Dropout(drop))

    # Adding the output layer
    regressor.add(Dense(62))

    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X.T, Y.T, epochs=25, batch_size=200)

    Y_hat = regressor.predict(X_test.T)
    Y_hat_train = regressor.predict(X.T)

    return Y_hat, Y_hat_train

#calculates accuracy of our model
def metric(Y_hat, Y):
    Y_hat_sign = np.sign(Y_hat.T)
    Y_sign = np.sign(Y)
    results = np.equal(Y_hat_sign, Y_sign)
    num_correct = np.sum(results)
    total = results.shape[0] * results.shape[1]
    return float(num_correct) / total

if __name__ == "__main__":
    entry()
