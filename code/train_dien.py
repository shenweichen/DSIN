# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K

from config import DIN_SESS_MAX_LEN, FRAC
from deepctr.models import DIEN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

if __name__ == "__main__":

    DIEN_NEG_SAMPLING = True
    FRAC = FRAC
    SESS_MAX_LEN = DIN_SESS_MAX_LEN
    dnn_feature_columns = pd.read_pickle('../model_input/dien_fd_' +
                        str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
    model_input = pd.read_pickle(
        '../model_input/dien_input_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
    label = pd.read_pickle('../model_input/dien_label_' +
                           str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')

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

    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 4096
    history_feature_list = ['cate_id', 'brand']
    TEST_BATCH_SIZE = 2 ** 14

    model = DIEN(dnn_feature_columns, history_feature_list,#v0.9.2
         gru_type="AUGRU", use_negsampling=DIEN_NEG_SAMPLING, dnn_hidden_units=(200, 80),
         dnn_activation='relu',
         att_hidden_units=(64, 16), att_activation="sigmoid", )

# def DIEN(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=16, #origin
#          gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(200, 80),
#          dnn_activation='relu',
#          att_hidden_units=(64, 16), att_activation="dice", att_weight_normalization=True,
#          l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary')

# def DIEN(dnn_feature_columns, history_feature_list,#v0.9.2
#          gru_type="AUGRU", use_negsampling=DIEN_NEG_SAMPLING, dnn_hidden_units=(200, 80),
#          dnn_activation='relu',
#          att_hidden_units=(64, 16), att_activation="sigmoid", )

    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy', ])
    if DIEN_NEG_SAMPLING:
        hist_ = model.fit(train_input, train_label, batch_size=BATCH_SIZE,
                          epochs=1, initial_epoch=0, verbose=1, )
        pred_ans = model.predict(test_input, TEST_BATCH_SIZE)
    else:
        hist_ = model.fit(train_input[:-3] + train_input[-1:], train_label, batch_size=BATCH_SIZE, epochs=1,
                          initial_epoch=0, verbose=1, )
        pred_ans = model.predict(
            test_input[:-3] + test_input[-1:], TEST_BATCH_SIZE)

    print()

    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
