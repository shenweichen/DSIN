# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K

from config import DIN_SESS_MAX_LEN, FRAC
from models import DIN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

if __name__ == "__main__":
    FRAC = FRAC
    SESS_MAX_LEN = DIN_SESS_MAX_LEN
    fd = pd.read_pickle('../model_input/din_fd_' +
                        str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
    model_input = pd.read_pickle(
        '../model_input/din_input_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
    label = pd.read_pickle('../model_input/din_label_' +
                           str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')

    sample_sub = pd.read_pickle(
        '../sampled_data/raw_sample_' + str(FRAC) + '.pkl')

    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    train_idx = sample_sub.loc[sample_sub.time_stamp <
                               1494633600, 'idx'].values
    test_idx = sample_sub.loc[sample_sub.time_stamp >=
                              1494633600, 'idx'].values

    train_input = [i[train_idx] for i in model_input]
    test_input = [i[test_idx] for i in model_input]
    train_label = label[train_idx]
    test_label = label[test_idx]

    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 4096

    sess_feature = ['cate_id', 'brand']
    TEST_BATCH_SIZE = 2 ** 14

    model = DIN(fd, sess_feature, embedding_size=4, att_activation='dice',
                att_weight_normalization=False, hist_len_max=sess_len_max, dnn_hidden_units=(200, 80),
                att_hidden_size=(64, 16,))

    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy', ])

    hist_ = model.fit(train_input[:], train_label,
                      batch_size=BATCH_SIZE, epochs=1, initial_epoch=0, verbose=1, )
    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

    print()
    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
