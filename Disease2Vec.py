#coding:utf-8
import tensorflow as tf

class Disease2Vec(object):

    # Neural network structure
    def __init__(self, input_dim, day_dim, diag_size, diag_dim, output_dim, L2=1e-8, win_size=1,
                 opt=tf.train.AdadeltaOptimizer(learning_rate=0.5), init_scale=0.01):
        self.input_dim = input_dim
        self.day_dim = day_dim
        self.diag_size = diag_size
        self.diag_dim = diag_dim
        self.output_dim = output_dim
        self.L2 = L2
        self.win_size = win_size
        self.hidden_dim = self.diag_dim + self.win_size * self.day_dim
        self.init_scale = init_scale

        self.keep_prob = tf.placeholder(tf.float32)

        # ori_d: RealValue Data; ori_d_s:Multihot Data
        self.ori_d = tf.placeholder(tf.float32, [None, self.input_dim])
        self.ori_d_s = tf.placeholder(tf.float32, [None, self.input_dim])
        self.w1 = tf.Variable(tf.random_normal([self.input_dim, self.day_dim],
                                               stddev=self.init_scale),dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([self.day_dim], dtype=tf.float32))
        self.w2 = tf.Variable(tf.random_normal([self.hidden_dim, self.input_dim],
                                               stddev=self.init_scale),dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([self.input_dim], dtype=tf.float32))
        self.diag = tf.Variable(tf.random_uniform([self.diag_size, self.diag_dim], 0.0, 1.0))
        self.diag = tf.nn.dropout(self.diag,self.keep_prob)

        self.d2diag = tf.placeholder(tf.int32, [None])
        self.mask = tf.placeholder(tf.float32, [None,1])

        self.hidden_d = tf.nn.relu(tf.add(tf.matmul(self.ori_d, self.w1), self.b1))
        self.hidden_d = tf.nn.dropout(self.hidden_d,self.keep_prob)

        self.d_agg = self._init_aggregate_day()

        zero_diag = tf.constant(0, shape=[1, self.diag_dim], dtype=tf.float32)

        zero_mask = tf.constant(0, shape=[1, 1], dtype=tf.float32)
        # mask_1 is used to filter the first day of every visit
        mask_1 = self.mask[:-1] * self.mask[1:]
        mask_1 = tf.concat([zero_mask, mask_1],0)
        self.u = (tf.concat([tf.gather(tf.concat([zero_diag, self.diag], 0), self.d2diag), self.d_agg], 1))*mask_1

        self.out_d = tf.nn.softmax(tf.matmul(tf.nn.relu(self.u), self.w2) + self.b2)

        self.ce = ( -self.ori_d_s * tf.log(self.out_d + self.L2) - (1. - self.ori_d_s) * tf.log(1. - self.out_d + self.L2) ) * mask_1

        self.loss = tf.reduce_sum(self.ce) / (tf.reduce_sum(mask_1) + self.L2)

        self.opt = opt.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # According to win_size to aggregate data
    def _init_aggregate_day(self):
        zero_day = tf.constant(0, shape=[1, self.day_dim], dtype=tf.float32)
        mask_i = self.mask
        for i in range(1, self.win_size + 1):  # i: 1-win_size
            mask_i = mask_i[:-1] * mask_i[1:]
            d_temp = mask_i * self.hidden_d[:-i]
            for j in range(i):
                d_temp = tf.concat([zero_day, d_temp], 0)
            if i == 1:
                d_agg = d_temp
            else:
                d_agg = tf.concat([d_temp, d_agg], 1)
        return d_agg

    def start_train(self, ori_d=None, ori_d_s=None, d2diag=None, mask=None,keep_prob=0.8):
        d_agg, out_d, loss, opt = self.sess.run((self.d_agg, self.out_d, self.loss, self.opt),
                    feed_dict={self.ori_d:ori_d, self.ori_d_s:ori_d_s, self.d2diag:d2diag, self.mask:mask,self.keep_prob:keep_prob})
        return d_agg, out_d, loss, opt


    def get_result(self, ori_d=None, ori_d_s=None, d2diag=None, mask=None,keep_prob=0.8):
        return self.sess.run((self.out_d, self.loss), feed_dict={self.ori_d:ori_d, self.ori_d_s:ori_d_s, self.d2diag:d2diag,
                                self.mask:mask,self.keep_prob:keep_prob})
