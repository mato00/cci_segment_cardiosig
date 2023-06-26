import argparse
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import sys
sys.path.append('../')

import scipy.io as sio
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import model_from_json
import wfdb

import ecg_preprocess as ep
import post_process as pp
from module.architecture4qrs import *
from module.tcn import TCN, tcn_full_summary

BATCH = 100

def load_ecg(file_path, database):
    if database == 'cpsc2019':
        sig = sio.loadmat(file_path)['ecg'].squeeze()
        fs = 500
        length = len(sig)
        lead_num = 1
    elif database == 'mitdb':
        sig, fields = wfdb.rdsamp(file_path)
        length = len(sig)
        fs = fields['fs']
        lead_num = np.shape(sig)[-1]
    elif database == 'incart':
        sig, fields = wfdb.rdsamp(file_path)
        length = len(sig)
        fs = fields['fs']
        lead_num = np.shape(sig)[-1]
    elif database == 'ludb':
        sig, fields = wfdb.rdsamp(file_path)
        length = len(sig)
        fs = fields['fs']
        lead_num = np.shape(sig)[-1]
    elif database == 'qt':
        sig, fields = wfdb.rdsamp(file_path)
        length = len(sig)
        fs = fields['fs']
        lead_num = np.shape(sig)[-1]
    else:
        print('The testing database is not in the range of unified testing platform!')

    return sig, length, fs, lead_num

def ngrams(data, length, fs):
    grams = []
    for i in range(0, length-fs*10, fs*6):
        grams.append(data[i: i+fs*10])
    return grams

def qrs_detect(database, data_path, ans_path):
    files = open(os.path.join(data_path, 'RECORDS'), 'r').read().splitlines()

    if not os.path.exists(ans_path):
        os.makedirs(ans_path)

    
    with open('./model_qrs/crnn_backbone/qrs_detector.json', 'r') as json_file:
        config= json_file.read()
    model = model_from_json(config, custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization})
    model.load_weights('./model_qrs/crnn_backbone/qrs_detector.h5')

    for k, file in enumerate(files):
        print(file)
        if database == 'cpsc2019':
            file_path = os.path.join(data_path, file+'.mat')
            ecg, length, fs, lead_num = load_ecg(file_path, database)
            
            ecg_test = ep.ecg_process(ecg, fs)
            ecg_test = np.expand_dims(ecg_test, 0)
        
            logits = model(ecg_test)
            logits_qrs = logits[:, :, 0]
            logits_qrs = np.squeeze(logits_qrs)
            logits_qrs = np.around(logits_qrs, decimals=2)

            logits_qrs = np.array([[j] * 4 for j in logits_qrs]).flatten()

            r_ans = pp.qrs_correct(logits_qrs)
            r_ans = r_ans.astype(float)
            r_ans *= (fs/250)
            r_ans = np.trunc(r_ans)

            np.save(os.path.join(ans_path, '{}.npy'.format(file)), r_ans)

        elif database == 'ludb':
            file_path = os.path.join(data_path, file)
            ecg, length, fs, lead_num = load_ecg(file_path, database)
            
            for j in range(lead_num):
                ecg_singleL = ecg[:, j]

                ecg_test = ep.ecg_process(ecg_singleL, fs)
                ecg_test = np.expand_dims(ecg_test, 0)

                logits = model(ecg_test)
                logits = np.squeeze(logits)
                logits = np.around(logits, decimals=2)

                logits = np.array([[j] * 4 for j in logits]).flatten()

                r_ans = pp.qrs_correct(logits)
                r_ans = r_ans.astype(float)
                r_ans *= (fs/250)
                r_ans = np.trunc(r_ans)

                np.save(os.path.join(ans_path, '{}_{}.npy'.format(file, str(j))), r_ans)

        else:
            file_path = os.path.join(data_path, file)
            ecg, length, fs, lead_num = load_ecg(file_path, database)

            for j in range(lead_num):
                ecg_singleL = ecg[:, j]
                ecg_batch = ngrams(ecg_singleL, length, fs)
                ecg_batch = np.array(ecg_batch)
                ecg_batch = ecg_batch[:, :]
                ecg_test = ep.ecg_process_batch(ecg_batch, fs)

                preds = []
                for i in range(len(ecg_test) // BATCH):
                    ecg_b = ecg_test[i*BATCH: (i+1)*BATCH]
                    logits = model(ecg_b)
                    logits = np.squeeze(logits)
                    logits = np.around(logits, decimals=2)

                    logits = np.reshape(logits, (BATCH, 625))
                    preds.extend(logits)

                if (len(ecg_test) - BATCH*(len(ecg_test)//BATCH)) > 0:
                    ecg_b = ecg_test[BATCH*(len(ecg_test)//BATCH): ]
                    logits = model(ecg_b)
                    logits = np.squeeze(logits)
                    logits = np.around(logits, decimals=2)

                    logits = np.reshape(logits, (len(ecg_test) - BATCH*(len(ecg_test)//BATCH), 625))
                    preds.extend(logits)

                ecg_last = ecg_singleL[-fs*10:, ]
                ecg_last = ep.ecg_process(ecg_last, fs)
                ecg_last = np.expand_dims(ecg_last, 0)

                logits_last = model(ecg_last)
                logits_last = np.squeeze(logits_last)
                logits_last = np.around(logits_last, decimals=2)

                gram_len = fs*6*(len(ecg_batch)-2) + 2*fs*8
                t_remain = (length - gram_len) / fs
                pred_remain = logits_last[-1*int((t_remain+2)*62.5): ]

                preds = np.array(preds)
                preds_init = preds[0, : 500]
                pred_post = preds[1: , 125: 500]
                preds = np.insert(pred_post, 0, preds_init)
                preds = np.insert(preds, -1, pred_remain)
                preds = np.array([[j] * 4 for j in preds]).flatten()

                r_ans = pp.qrs_correct(preds)
                r_ans = r_ans.astype(float)
                r_ans *= (fs/250)
                r_ans = np.trunc(r_ans)

                np.save(os.path.join(ans_path, '{}_{}.npy'.format(file, str(j))), r_ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--database_type',
                        help='database type')
    parser.add_argument('-d',
                        '--data_path',
                        help='path saving test record file')
    parser.add_argument('-r',
                        '--result_save_path',
                        help='path saving QRS-complex location results')

    args = parser.parse_args()

    qrs_detect(database=args.database_type, data_path=args.data_path, ans_path=args.result_save_path)
