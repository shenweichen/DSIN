# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K

from config import DSIN_SESS_COUNT, DSIN_SESS_MAX_LEN, FRAC
from deepctr.models import DSIN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

if __name__ == "__main__":
    SESS_COUNT = DSIN_SESS_COUNT
    SESS_MAX_LEN = DSIN_SESS_MAX_LEN

    dnn_feature_columns = pd.read_pickle('../model_input/dsin_fd_' +
                                         str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    model_input = pd.read_pickle(
        '../model_input/dsin_input_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    label = pd.read_pickle('../model_input/dsin_label_' +
                           str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')

    sample_sub = pd.read_pickle(
        '../sampled_data/raw_sample_' + str(FRAC) + '.pkl')

    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    train_idx = sample_sub.loc[sample_sub.time_stamp <
                               1494633600, 'idx'].values
    test_idx = sample_sub.loc[sample_sub.time_stamp >=
                              1494633600, 'idx'].values

    train_input = {k: v[train_idx] for k, v in model_input.items()}
    test_input = {k: v[test_idx] for k, v in model_input.items()}

    train_label = label[train_idx]
    test_label = label[test_idx]

    sess_count = SESS_COUNT
    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 4096

    sess_feature_list = ['cate_id', 'brand']
    TEST_BATCH_SIZE = 2 ** 14

    # model = DSIN(fd, sess_feature, sess_max_count=sess_count, dnn_hidden_units=(200, 80), att_head_num=8,
    #              att_embedding_size=1, bias_encoding=True,dnn_activation='relu')

    model = DSIN(dnn_feature_columns, sess_feature_list, sess_max_count=sess_count, bias_encoding=False,
                 att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='relu',
                 )

    # def DSIN(feature_dim_dict, sess_feature_list, embedding_size=8, sess_max_count=5, sess_len_max=10, bias_encoding=False,
    #          att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='sigmoid', dnn_dropout=0,
    #          dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, init_std=0.0001, seed=1024, task='binary',
    #          ):#origin

    # def DSIN(dnn_feature_columns, sess_feature_list, sess_max_count=sess_count, bias_encoding=True,
    #          att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='relu',
    #          ):#v0.9.2

    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy', ])

    hist_ = model.fit(train_input, train_label, batch_size=BATCH_SIZE,
                      epochs=1, initial_epoch=0, verbose=1, )

    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

    print()
    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
