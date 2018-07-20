####paper: https://blog.csdn.net/a819825294/article/details/71215538


import linecache
import numpy as np
import tensorflow as tf
import time
import math
import os
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from preprocess_ml_1m_TEST2 import *

#### all parameter
batch_size = 100
emb_size = 128
max_window_size = 70

learning_rate = 0.0001
training_epochs = 300
display_step = 1
y_size = 25
# Network Parameters
n_hidden_1 = 128 # 1st layer number of features
# n_hidden_2 = 256 # 2nd layer number of features

# init_data(train_file)
n_classes = len(movie_line)
# train_lst = linecache.getlines(train_file)
print("Class Num: ", n_classes)

### store y label for training set
y_label = {}

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([emb_size, n_hidden_1])),
    # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    #x = tf.nn.dropout(x, 0.8)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #dlayer_1 = tf.nn.dropout(layer_1, 0.5)
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # return out_layer
    return layer_1

#####embedding
embedding = {
    'input':tf.Variable(tf.random_uniform([n_classes, emb_size], -1.0, 1.0))
    # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
}

##### initialize batch parameter
word_num = tf.placeholder(tf.float32, shape=[None, 1])
x_batch = tf.placeholder(tf.int32, shape=[None, max_window_size])   ###max_window_size
y_batch = tf.placeholder(tf.float32, shape=[None, 3883]) ###one-hot？？？



input_embedding = tf.nn.embedding_lookup(embedding['input'], x_batch)
project_embedding = tf.div(tf.reduce_sum(input_embedding, 1),word_num)

check_op = tf.add_check_numerics_ops()




# Construct model
pred = multilayer_perceptron(project_embedding, weights, biases)

# Construct the variables for the NCE loss
score = tf.matmul(pred, tf.transpose(embedding['input']))
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = score, labels = y_batch)

cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()

out_layer = tf.nn.sigmoid(score)

#### read data function
def read_data(pos, batch_size, data_lst, neg_lst):  # data_lst = u_mid_pos: {use:(mid,rate)}
    """
    :param pos:
    :param batch_size:
    :param data_lst:
    :return: returns a set of numpy arrays, which will be fed into tensorflow placeholders
    """
    batch = {}
    i = pos
    for key, value in data_lst.copy().items():
        batch.update({key: value})
        del [data_lst[key]]
        pos += 1
        if (pos >= i + batch_size):
            break

    x = np.zeros((batch_size, max_window_size))
    y = np.zeros((batch_size, n_classes), dtype=int)


    word_num = np.zeros((batch_size))

    line_no = 0

    for key, value in batch.items():
        col_no_x = 0
        col_no_y = 0
        # print(line_no)
        for i in value:
            # update y
            ####one hot encoding for y has five labels
            if (col_no_y < y_size):
                index = int(i[0])
                y[line_no][index] = 1
                col_no_y += 1
                ## store in y_label:
                y_label.setdefault(key,set()).add(i[0])


            # update x
            ###other use as embedding look up for x
            else:
                index = int(i[0])
                # y[line_no][index] = 1
                x[line_no][col_no_x] = i[0]
                col_no_x += 1

            if col_no_x >= max_window_size:
                break

####add negative samples:  set one hot encoding for negative sample = -1
        if key in neg_lst:
            count = 0
            for i in neg_lst[key]:
                index = int(i[0]) - 1
                y[line_no][index] = -1
                if(count > y_size*3):
                    break
                count = count + 1


        word_num[line_no] = col_no_x
        line_no += 1

    return x, y, word_num.reshape(batch_size, 1)


def read_data_test(pos, batch_size, data_lst, neg_lst):  # data_lst = u_mid_pos: {use:(mid,rate)}
    """
    :param pos:
    :param batch_size:
    :param data_lst:
    :return: returns a set of numpy arrays, which will be fed into tensorflow placeholders
    """
    batch = {}
    i = pos
    for key, value in data_lst.copy().items():
        batch.update({key: value})
        del [data_lst[key]]
        pos += 1
        if (pos >= i + batch_size):
            break

    x = np.zeros((batch_size, max_window_size))
    y = np.zeros((batch_size, n_classes), dtype=int)
    y_label_test = np.zeros((batch_size, y_size), dtype=int)


    word_num = np.zeros((batch_size))

    line_no = 0

    for key, value in batch.items():
        col_no_x = 0
        col_no_y = 0
        col_no_y_label = 0
        for i in value:
            # update y
            ####one hot encoding for y has five labels
            index = int(i[0])
            y[line_no][index] = 1
            col_no_y += 1

        # update x
        for index in u_mid_pos[key]:
            x[line_no][col_no_x] = index[0]
            col_no_x += 1
            if col_no_x >= max_window_size:
                break

        ##update y-label
        for u in y_label[key]:
            y_label_test[line_no][col_no_y_label] = str(u)
            col_no_y_label += 1

        ####add negative samples:  set one hot encoding for negative sample = -1
        if key in neg_lst:
            count = 0
            for i in neg_lst[key]:
                index = int(i[0]) - 1
                y[line_no][index] = -1
                if (count > col_no_y * 3):
                    break
                count = count + 1

        word_num[line_no] = col_no_x
        line_no += 1


    return x, y, word_num.reshape(batch_size, 1), y_label_test


###################################### Test model
    # _, top = tf.nn.top_k(out_layer, k = 11)
    # correct_prediction = tf.equal(top, y_batch)
    # Calculate accuracy
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
def test():
    test_lst = u_mid_pos_test
    total_batch = int(len(test_lst) / batch_size)

    ##### top k accuracy:
    k = 30
    rec_count = 0
    hit = 0
    test_count = 0
    # all_rec_movies = set()
    avg_cost = 0

    for i in range(total_batch):
        copy = u_mid_pos_test.copy()
        x, y, word_number, y_label_test = read_data_test(i * batch_size, batch_size, copy, u_mid_neg)
        out_score = out_layer.eval({x_batch: x, word_num: word_number})

        ### cost
        c = cost.eval({x_batch: x, word_num: word_number, y_batch: y })


        avg_cost += c / total_batch
        ## calculate recall and precision


        for row_x, row_out, row_y, row_y_label in zip(x, out_score,y, y_label_test):
            ## set the training labels' prob as 0
            for col in row_x:
                row_out[int(col)] = 0
            for col in row_y_label:
                row_out[int(col)] = 0

            ##get top k index
            top_k = np.argsort(row_out)[::-1][:k]
            # print("predict", top_k)
            # print("real_y",  np.where(row_y == 1))
            # print("real_x", row_x)
            for index in top_k:
                if(row_y[index] == 1):
                    hit += 1
            rec_count += k
            test_count += y_size
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    print("validation cost", avg_cost)
    # coverage = len(all_rec_movies) / (1.0 * movie_count)
    print('precision=%.4f\trecall=%.4f\n' %
            (precision, recall))









#########run

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

        for i in range(total_batch):
            x, y, word_number = read_data(i * batch_size, batch_size, copy, u_mid_neg)
            # print(x)
            # print(word_number)
            # print(y)
            _, c, a = sess.run([optimizer, cost, check_op],
                            feed_dict={x_batch: x, y_batch: y, word_num: word_number})

            # print("loss", l)
            avg_cost += c / total_batch



        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost))

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