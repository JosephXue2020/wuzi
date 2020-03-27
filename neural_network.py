# coding = 'utf8'
# date: 2019.11.01


import os
import sys
import random

import numpy as np
import tensorflow as tf

import board_wuzi


'''
Construct the class of neural network. Two kinds of NNs are constructed, CNN and ResNet. The CNN only has 3
convolution layers for simpler.
'''


# 管理模型保存路径，包括训练模型、对比模型等
class ModelLoader:
    def __init__(self):
        # 下面的语法结构，在终端只给出相对路径，所以会出错
        # self.current_path = sys.argv[0]
        # self.file_dir, _ = os.path.split(self.current_path)

        # self.file_dir = r'D:\PycharmProjects\wuzi (base on Alphago Zero) -11x11'  # 这个是为了在cmd终端运行用的

        # 下面这个终端可用
        self.file_dir = os.getcwd()

    def model_dir(self, folder='\\model\\'):
        model_dir = self.file_dir + folder
        return model_dir

    def model_path(self, model='my_model.ckpt'):
        model_dir = self.model_dir()
        model_path = model_dir + model
        return model_path

    def model_path_compared(self, model='\\model_compared\\my_model.ckpt'):
        model_path = self.file_dir + model
        return model_path

    def training_data_path(self, file='training_data.json'):
        file_path = self.model_dir() + file
        return file_path

    def losses_path(self, file='losses.cvs'):
        file_path = self.model_dir() + file
        return file_path


model_loader = ModelLoader()


class CNN:
    def __init__(self, **kwargs):
        # 设置网络的超参数：
        self.batch_size = kwargs.get('batch_size', 30)
        self.image_size = kwargs.get('board_size')
        self.channel = kwargs.get('channel')  # 最后一层表征走子方，其余的是黑白交替盘面
        self.filter_size = 3
        self.common_layers = [128, 128, 64]  # 列表长度代表卷积层数，值代表输出channel
        self.policy_layers = [16, 256, self.image_size**2]
        self.value_layers = [8, 128, 1]
        self.learning_rate = 0.0001
        tf.reset_default_graph()
        self.x_input = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, self.channel),
                                      name='x_input')
        self.y_output = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size), name='y_output')
        self.z_output = tf.placeholder(tf.float32, shape=(None), name='z_output')
        self.is_training = tf.placeholder(tf.bool, name='is_training')  # 为dropout准备
        self.weight_decay = 0.0004
        # 下面这句定义regularizer在pycharm中运行有问题，所以还是不要用了
        # self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
        self.construction_phase()

    def construction_phase(self):
        # 卷积层
        tensor = self.x_input
        for item in self.common_layers:
            tensor = self.conv_layer(tensor, self.filter_size, item, activation='relu')

        # policy分支。首先1x1卷积，降维
        policy_tensor = self.conv_layer(tensor, 1, self.policy_layers[0], activation='relu')
        policy_tensor_shape = policy_tensor.get_shape().as_list()
        policy_tensor = tf.reshape(policy_tensor,
                                   shape=[-1, policy_tensor_shape[1]*policy_tensor_shape[2]*policy_tensor_shape[3]])
        for item in self.policy_layers[1:-1]:
            policy_tensor = self.full_layer(policy_tensor, item, 'relu')
        # 注意：最后一层不能接sigmoid
        logits_p = self.full_layer(policy_tensor, self.policy_layers[-1], dropout=False)
        outputs_p = tf.nn.softmax(logits_p)
        self.outputs_p = tf.reshape(outputs_p, shape=[-1, self.image_size, self.image_size])

        # value分支。首先1x1卷积，降维
        value_tensor = self.conv_layer(tensor, 1, self.value_layers[0], activation='relu')
        value_tensor_shape = value_tensor.get_shape().as_list()
        value_tensor = tf.reshape(value_tensor,
                                  shape=[-1, value_tensor_shape[1]*value_tensor_shape[2]*value_tensor_shape[3]])
        for item in self.value_layers[1:-1]:
            value_tensor = self.full_layer(value_tensor, item, 'relu')
        # 最后一层不要dropout
        self.outputs_v = self.full_layer(value_tensor, self.value_layers[-1], activation='tanh', dropout=False)

        # 定义loss
        y_output_reshaped = tf.reshape(self.y_output, shape=[-1, self.image_size**2])
        base_loss = tf.reduce_mean((self.z_output - self.outputs_v)**2) \
               + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_output_reshaped, logits=logits_p))
        reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.add_n(reg_set)
        self.loss = base_loss + reg_loss
        # self.loss = base_loss
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # 加入以下代码，会更新mean和variance，以便预测时调用
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def conv_layer(self, x_input, filter_size, out_channel, activation=None):
        if activation == 'relu':
            activation = tf.nn.relu
        if activation == 'sigmoid':
            activation = tf.nn.sigmoid
        if activation == 'tanh':
            activation = tf.nn.tanh
        with tf.name_scope('conv_layer'):
            x_input_shape = x_input.shape.as_list()
            image_size_in, in_channel = x_input_shape[1], x_input_shape[-1]
            if image_size_in < filter_size:
                filter_size = image_size_in

            # 手动计算l2_loss，并加入tf.GraphKeys.REGULARIZATION_LOSSES。BN时不用bias
            w_conv = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel,
                                                                        out_channel], mean=0, stddev=1), name='w_conv')
            l2_reg = tf.reduce_sum(tf.multiply(w_conv, w_conv)) * self.weight_decay / 2
            # 下面这句有问题，不要用了
            # l2_reg = self.regularizer(w_conv)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_reg)
            # b_conv = tf.Variable([0.0], name='b_conv')
            z = tf.nn.conv2d(x_input, w_conv, strides=[1, 1, 1, 1], padding='SAME')
            # z += b_conv

            # 在tf.layers.conv2d中封装的，可以利用kernel_initializer，但应用中似乎有问题
            # z = tf.layers.conv2d(x_input, out_channel, [filter_size, filter_size], strides=[1, 1],
            #                 use_bias=not self.batch_normalization_bool, padding='SAME',
            #                 kernel_regularizer=self.regularizer)

            if activation:
                z = tf.layers.batch_normalization(z, training=self.is_training)
                conv_out = activation(z)
            else:
                conv_out = z
            return conv_out

    def full_layer(self, x_input, n_hidden, activation=None, dropout=True):
        if activation == 'relu':
            activation = tf.nn.relu
        if activation == 'sigmoid':
            activation = tf.nn.sigmoid
        if activation == 'tanh':
            activation = tf.nn.tanh
        with tf.name_scope('full_layer'):
            # 加入regularizer。这个封装也有问题
            # z = tf.layers.dense(x_input, n_hidden, use_bias=not self.batch_normalization_bool,
            #                    kernel_regularizer=self.regularizer)

            # 手动构建全连接
            input_shape = x_input.get_shape().as_list()
            w_full = tf.Variable(tf.truncated_normal([input_shape[-1], n_hidden], mean=0., stddev=1.), name='w_full')
            l2_reg = tf.reduce_sum(tf.multiply(w_full, w_full)) * self.weight_decay / 2
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_reg)
            # b_full = tf.Variable([0.], name='b_full')
            z = tf.matmul(x_input, w_full)

            z = tf.layers.batch_normalization(z, training=self.is_training)
            if activation:
                # z = tf.layers.batch_normalization(z, training=self.is_training)
                full_out = activation(z)
                if dropout:
                    full_out = tf.layers.dropout(full_out, rate=0.5, training=self.is_training)
            else:
                full_out = z
        return full_out

    def train(self, data_loader, training_time, start_over=False):
        model_dir = model_loader.model_dir()
        model_path = model_loader.model_path()
        is_training = True
        if start_over or not os.listdir(model_dir):
            with tf.Session() as sess:
                sess.run(self.init)
                for _ in range(training_time):
                    x_batch, y_batch, z_batch = data_loader.get_next_batch()
                    feed_dict = {self.x_input: x_batch, self.y_output: y_batch, self.z_output: z_batch,
                                 self.is_training: is_training}
                    sess.run(self.train_op, feed_dict=feed_dict)
                self.saver.save(sess, model_path)
        else:
            with tf.Session() as sess:
                self.saver.restore(sess, model_path)
                for _ in range(training_time):
                    x_batch, y_batch, z_batch = data_loader.get_next_batch()
                    feed_dict = {self.x_input: x_batch, self.y_output: y_batch, self.z_output: z_batch,
                                 self.is_training: is_training}
                    sess.run(self.train_op, feed_dict=feed_dict)
                print('Success to train one time.')
                self.saver.save(sess, model_path)

    def predict(self, img):
        # 对输入数据增强。predict_v取平均值；predict_p取第一个就好
        x_batch = np.array(DataLoader.state_augment(img))
        feed_dict = {self.x_input: x_batch, self.is_training: False}
        predict_p = self.sess_pred.run(self.outputs_p, feed_dict=feed_dict)
        predict_v = self.sess_pred.run(self.outputs_v, feed_dict=feed_dict)
        predict_p = predict_p[0, :, :]
        predict_v = np.mean(predict_v)
        return predict_p, predict_v

    # 避免频繁打开关闭session
    def predict_session_open(self, flag=None):
        model_dir = model_loader.model_dir()
        model_path = model_loader.model_path()
        model_path_compared = model_loader.model_path_compared()
        self.sess_pred = tf.Session()
        if not flag:  # 为训练阶段，读取当前训练模型
            if not os.listdir(model_dir):
                self.sess_pred.run(self.init)
            else:
                self.saver.restore(self.sess_pred, model_path)
        else:
            self.saver.restore(self.sess_pred, model_path_compared)

    def predict_session_close(self):
        self.sess_pred.close()

    def validation(self, data_loader):
        model_dir = model_loader.model_dir()
        if os.listdir(model_dir):
            model_path = model_loader.model_path()
            with tf.Session() as sess:
                self.saver.restore(sess, model_path)
                x_batch, y_batch, z_batch = data_loader.get_next_batch()
                feed_dict = {self.x_input: x_batch, self.y_output: y_batch, self.z_output: z_batch, self.is_training: False}
                loss = sess.run(self.loss, feed_dict=feed_dict)
                return loss


class DataLoader:
    def __init__(self, dataset, batch_size=None):
        self.data = list(dataset)
        if not batch_size:
            self.batch_size = len(self.data)
            self.batch_generator = self.batch_generator()
        else:
            self.batch_size = batch_size
            self.data_augment()
            self.batch_generator = self.batch_generator()

    def get_next_batch(self):
        return next(self.batch_generator)

    def batch_generator(self):
        batches = self.get_batches()
        idx = list(range(len(batches)))
        while True:
            i = idx[0]
            idx = idx[1:]
            idx.append(i)
            yield batches[i]

    # 这里应该可以优化，节约内存
    def get_batches(self):
        random.shuffle(self.data)
        batches = []
        batch_num = int(len(self.data)/self.batch_size)
        for i in range(batch_num):
            seq = self.data[i:i+self.batch_size]
            seq_state = []
            seq_prob = []
            seq_z = []
            for item in seq:
                seq_state.append(item['state'])
                seq_prob.append(item['prob'])
                seq_z.append(item['z'])
            state_arr = np.array(seq_state)
            prob_arr = np.array(seq_prob)
            z_arr = np.array(seq_z).reshape(self.batch_size, 1)
            # z_arr = np.array(seq_z)
            batches.append([state_arr, prob_arr, z_arr])
        return batches

    def data_augment(self):
        data = []
        for item in self.data:
            state = item.get('state')
            states_aug = DataLoader.state_augment(state)
            prob = item.get('prob')
            probs_aug = DataLoader.state_augment(prob)
            for state_aug, prob_aug in zip(states_aug, probs_aug):
                element = item.copy()
                element['state'] = state_aug
                element['prob'] = prob_aug
                data.append(element)
        self.data = data

    @staticmethod
    def state_augment(arr):
        arr = np.array(arr)
        arrs = []
        arrs.append(arr)
        arrs.append(DataLoader.rotate_left_half_pi(arr))
        arrs.append(DataLoader.rotate_right_half_pi(arr))
        arrs.append(DataLoader.rotate_pi(arr))
        arrs_all = arrs.copy()
        for item in arrs:
            arrs_all.append(DataLoader.flip_pi(item))
        return arrs_all

    @staticmethod
    def rotate_left_half_pi(arr):
        length = len(arr.shape)
        if length == 2:
            new_arr = arr.transpose((1, 0))
            new_arr = new_arr[::-1, :]
        elif length == 3:
            new_arr = arr.transpose((1, 0, 2))
            new_arr = new_arr[::-1, :, :]
        else:
            raise ValueError('The rotate function can only treat dim of 2 and 3.')
        return new_arr

    @staticmethod
    def rotate_right_half_pi(arr):
        length = len(arr.shape)
        if length == 2:
            new_arr = arr.transpose((1, 0))
            new_arr = new_arr[:, ::-1]
        elif length == 3:
            new_arr = arr.transpose((1, 0, 2))
            new_arr = new_arr[:, ::-1, :]
        else:
            raise ValueError('The rotate function can only treat dim of 2 and 3.')
        return new_arr

    @staticmethod
    def rotate_pi(arr):
        length = len(arr.shape)
        if length == 2:
            new_arr = arr[::-1, ::-1]
        elif length == 3:
            new_arr = arr[::-1, ::-1, :]
        else:
            raise ValueError('The rotate function can only treat dim of 2 and 3.')
        return new_arr

    @staticmethod
    def flip_pi(arr):
        length = len(arr.shape)
        if length == 2:
            new_arr = arr[::-1, :]
        elif length == 3:
            new_arr = arr[::-1, :, :]
        else:
            raise ValueError('The rotate function can only treat dim of 2 and 3.')
        return new_arr


# 以下是对网络的test
if __name__ == '__main__':
    size = 6
    n_in_line = 4
    channel = 5
    img = np.zeros([size, size, 5])
    img[2,2]=1.
    prob = np.ones([size, size]) / size * size
    data_one = {'state': img, 'z': -1, 'prob': prob}
    # img2 = img.copy()
    # img2[5, 5, 0] = 1
    # img2[:, :, -1] = 1
    # img3 = img.copy()
    # img3[5, 6, 0] = 1
    # img3[5, 5, 1] = 1
    #
    # img4 = img.copy()
    # img4[5, 3:7, 0] = 1
    # img4[4, 3:7, 1] = 1
    # img4[5, 3:6, 2] = 1
    # img4[4, 3:6, 3] = 1
    # img4[:, :, -1] = -1
    #
    # imgf = img.copy()
    # imgf[4, 3:8, 0] = 1
    # imgf[5, 3:7, 1] = 1
    # imgf[4, 3:7, 2] = 1
    # imgf[5, 3:6, 3] = 1
    # imgf[:, :, -1] = 1

    #board = board_wuzi.Board(imgf, size=size, channel=channel_, n_in_line=n_in_line)
    cnn = CNN(board_size=size, channel=channel)

    cnn.predict_session_open()
    p, v = cnn.predict(img)
    # p2, v2 = cnn.predict(img2)
    # p3, v3 = cnn.predict(img3)
    # p4, v4 = cnn.predict(img3)
    # pf, vf = cnn.predict(imgf)
    cnn.predict_session_close()
    print(p)
    print('Sum of p is:', np.sum(p))
    print(v)
    # print(p4)
    # print('Sum of pf is:', np.sum(pf))
    # print(v, v2, v3, v4, vf)

    # print(board.state[:, :, 0])
    # print('The last move of imgf is:', board.last_move)
    # print('The player of imgf is:', board.get_player())
    # print('The winner of imgf is:', board.get_a_winner())
    #
    # move = [4, 8]
    # next_state = board.get_next_state_by_move(move)
    # print(next_state[:, :, -1])
