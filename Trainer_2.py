####paper: https://blog.csdn.net/a819825294/article/details/71215538


import linecache
import numpy as np
import tensorflow as tf
import time
import math
import os
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from preprocess_ml_1m import *


batch_size = 100
emb_size = 128
max_window_size = 50

learning_rate = 0.001
training_epochs = 50
display_step = 1

# Network Parameters
n_hidden_1 = 128 # 1st layer number of features
#n_hidden_2 = 30 # 2nd layer number of features

# init_data(train_file)
n_classes = len(movie_line)
# train_lst = linecache.getlines(train_file)
print("Class Num: ", n_classes)


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([emb_size, n_hidden_1])),
    # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    #x = tf.nn.dropout(x, 0.8)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, 0.5)
    # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # return out_layer
    # return layer_1

#####embedding
embedding = {
    'input':tf.Variable(tf.random_uniform([n_classes, emb_size], -1.0, 1.0))
    # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
}


word_num = tf.placeholder(tf.float32, shape=[None, 1])
x_batch = tf.placeholder(tf.int32, shape=[None, max_window_size])   ###max_window_size
y_batch = tf.placeholder(tf.int32, shape=[None, 5]) ###one-hot？？？

input_embedding = tf.nn.embedding_lookup(embedding['input'], x_batch)
project_embedding = tf.div(tf.reduce_sum(input_embedding, 1),word_num)


check_op = tf.add_check_numerics_ops()


# Construct model
pred = multilayer_perceptron(project_embedding, weights, biases)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([n_classes, n_hidden_1],
                        stddev=1.0 / math.sqrt(n_hidden_1)))
nce_biases = tf.Variable(tf.zeros([n_classes]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=y_batch,
                     inputs=pred,
                     num_sampled=5,
                     num_true = 5,
                     num_classes=n_classes))

cost = tf.reduce_sum(loss) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
out_layer = tf.matmul(pred, tf.transpose(nce_weights)) + nce_biases

init = tf.global_variables_initializer()


def read_data(pos, batch_size, data_lst): #data_lst = u_mid_pos: {use:(mid,rate)}
    """
    :param pos:
    :param batch_size:
    :param data_lst:
    :return: returns a set of numpy arrays, which will be fed into tensorflow placeholders
    """
    batch = {}
    # print("pos", pos)
    i = pos
    for key, value in data_lst.copy().items():
        batch.update({key: value})
        del[data_lst[key]]
        pos += 1
        if (pos >= i + batch_size):
            break



    x = np.zeros((batch_size, max_window_size))
    y = np.zeros((batch_size, 5),dtype=int)

    word_num = np.zeros((batch_size))

    line_no = 0

    for key,value in batch.items():
        col_no_x = 0
        col_no_y = 0

        #print(line_no)
        for i in value:
            # update y
            ####make sure the len is 11
            if(col_no_y < 5):
                y[line_no][col_no_y] = i[0]
                col_no_y += 1
            # update x
            else:
                x[line_no][col_no_x] = i[0]
                col_no_x += 1

            if col_no_x >= max_window_size:
                break

        word_num[line_no] = col_no_x
        line_no += 1

    return x, y, word_num.reshape(batch_size, 1)


###################################### Test model
    # _, top = tf.nn.top_k(out_layer, k = 11)
    # correct_prediction = tf.equal(top, y_batch)
    # Calculate accuracy
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

def test():
    test_lst = u_mid_pos_test
    total_batch = int(len(test_lst) / batch_size)

    final_accuracy = 0
    for i in range(total_batch):
        copy = u_mid_pos_test.copy()
        x, y, word_number = read_data(i * batch_size, batch_size, copy)
        _,top = tf.nn.top_k(out_layer, k=100)
        top = top.eval({x_batch: x, y_batch: y, word_num: word_number})
        cost0 = sess.run([cost],
                           feed_dict={x_batch: x, y_batch: y, word_num: word_number})
        print(cost0)
        # out_layer = out_layer.eval({x_batch: x, y_batch: y, word_num: word_number})
        # print(out_layer)
        # print(out_layer[1])
        # print(top[1])
        # print(y[1])
        # print(out_layer[2])
        # print(top[2])
        # print(y[2])

        ###accuracy
        accuracy = np.zeros((batch_size, 1))
        index_row = 0

        for i,j in zip(top, y):
            score = 0

            # print("........")

            # print("////////")
            for col_i in i:
                if col_i in j:
                    score += 1

            accuracy[index_row] = score / 20
            index_row += 1

        print(accuracy)
        batch_accuracy = np.mean(accuracy)
        final_accuracy += batch_accuracy
    print("Final Accuracy: ", final_accuracy * 20 / total_batch)





#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    start_time = time.time()
    total_batch = int(len(u_mid_pos) / batch_size)
    print("total_batch of training data: ", total_batch)
    for epoch in range(training_epochs):
        avg_cost = 0.
        copy = u_mid_pos.copy()
        print(len(u_mid_pos))
        for i in range(total_batch):
            x, y, word_number = read_data(i * batch_size, batch_size, copy)
            # print(word_number)
            # print(y)
            _, c, a = sess.run([optimizer, cost, check_op],
                            feed_dict={x_batch: x, y_batch: y, word_num: word_number})

            avg_cost += c / total_batch


        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost))

        ## call test:
        test()










"""
Typical project configuration using TensorFlow: 
1. model.py
    class MyModel(object):
        def __init__(self, params):
            # tensorflow nodes that you wish to be accessible from outside (interface nodes)
            self.optimizer = ...
            self.loss = None
            self.x_batch = tf.placeholder(...)
            pass
            
        def my_model(self, params, ...):
            building tensorflow computation graph
           
2. data_util.py
    class DataLoader:
        ...
        # takes care of loading data
        def generator():
            ...
            
3. train.py
    tf.flags.DEFINE_string(...)
    def main(_):     
        pass
    
    if __name__ == '__main__':
        tf.app.run()
"""