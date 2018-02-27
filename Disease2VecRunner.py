#coding:utf-8
import tensorflow as tf
import numpy as np
from Disease2Vec import Disease2Vec
import time

config = {}
starttime = time.time()
lasttime = time.time()

# Setting configuration
def init_config():
    print('init_config')
    config['input_dim'] = 4216
    config['day_size'] = 4496230
    config['day_dim'] = 200
    config['diag_size'] = 7232
    config['diag_dim'] = 150
    config['output_dim'] = 4216
    config['win_size'] = 2
    config['max_epoch'] = 20
    config['batch_size'] = 128
    config['topk'] = 30
    return config

# According to "d2bow_batch" to generate training data which dimensionality is config.input_dim  in every batch
def partial_data_generate(d2bow_batch):
    # print('partial_data_generate')
    ori_days = np.zeros((len(d2bow_batch), config['input_dim']), dtype='float32')
    ori_days_simple = np.zeros((len(d2bow_batch), config['input_dim']), dtype='float32')
    for i in range(len(d2bow_batch)):
        db = d2bow_batch[i]
        if db == -1:
            continue
        for wc in db:
            word = wc[0]
            count = wc[1]
            ori_days[i][word] = count * 1.0
            ori_days_simple[i][word] = 1
    return ori_days, ori_days_simple

def model_train(disease2vec, save_path, d2bow, d2diag, mask):
    print('model train')

    tt_sep = int(len(d2bow) * 0.8)
    while d2diag[tt_sep] != 0:
        tt_sep += 1

    d2bow_train = d2bow[0:tt_sep]
    d2bow_test = d2bow[tt_sep+1:len(d2bow)]
    d2diag_train = d2diag[0:tt_sep]
    d2diag_test = d2diag[tt_sep+1:len(d2bow)]
    mask_train = mask[0:tt_sep]
    mask_test = mask[tt_sep+1:len(d2bow)]

    for epoch in range(config['max_epoch']):
        starttime = time.time()
        avg_loss = 0.0

        iter = int(np.ceil(float(len(d2diag_train)) / config['batch_size']))

        # Consider days of a visit are cut by batch
        last_end = -1

        # Train
        print('------------training----------------')
        print('***Epoch %d***' % (epoch))
        for idx in range(iter):
            start = last_end + 1
            end = (idx + 1) * config['batch_size']
            while end < len(d2diag_train) and d2diag_train[end] != 0:  # Until the end of visit
                end += 1
            last_end = end
            if start >= end or start >= len(d2diag_train):
                continue

            d2bow_batch = d2bow_train[start:end]
            d2diag_batch = d2diag_train[start:end]
            mask_batch = mask_train[start:end]

            ori_days_batch,ori_days_simple_batch  = partial_data_generate(d2bow_batch)
            d_agg, out_d, loss, opt = disease2vec.start_train(ori_d = ori_days_batch, ori_d_s = ori_days_simple_batch,
                                                              d2diag=d2diag_batch, mask=mask_batch)
            avg_loss += loss * (len(d2diag_batch))

        avg_loss = avg_loss / len(d2diag_train)
        print("Train Length = " + str(len(d2bow_train)))
        print('loss: %f, takes: %f' % (avg_loss, time.time() - starttime))


        # Test
        print('------------test----------------')
        iter_test = int(np.ceil(float(len(d2diag_test)) / config['batch_size']))
        for idx in range(iter_test):
            start = (idx) * config['batch_size']
            end = (idx + 1) * config['batch_size']

            d2bow_batch_test = d2bow_test[start:end]
            d2diag_batch_test = d2diag_test[start:end]
            mask_batch_test = mask_test[start:end]

            ori_days_batch_test, ori_days_simple_batch_test = partial_data_generate(d2bow_batch_test)
            out_d_test, loss_test = disease2vec.get_result(
                ori_d=ori_days_batch_test, ori_d_s=ori_days_simple_batch_test, d2diag=d2diag_batch_test, mask=mask_batch_test)

            avg_loss = loss_test  * len(d2diag_batch_test)

        avg_loss = avg_loss / len(d2diag_test)
        print('loss_test: %f' %(avg_loss) )

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess=disease2vec.sess, save_path=save_path + '-epoch-' + str(epoch), global_step=config['max_epoch'])
        print('model saved')

def main(disease2vec_saver_file,d2bow,d2diag,mask):

    init_config()
    print("Start Train")
    disease2vec = Disease2Vec(input_dim=config['input_dim'], day_dim=config['day_dim'],
                    diag_size=config['diag_size'], diag_dim=config['diag_dim'],
                    output_dim=config['output_dim'], win_size=config['win_size'])

    model_train(disease2vec, disease2vec_saver_file, d2bow, d2diag, mask)

if __name__=='__main__':

    d2bow = [[]]
    d2diag = []
    mask = [[]]

    main("/",d2bow,d2diag,mask)