import argparse
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import sys
sys.path.append('../')

import librosa
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import model_from_json

import hs_preprocess as hsp
import post_process as pp
from module.architecture4hs import *


def ngrams(data, length, fs):
    grams = []
    for i in range(0, length-fs*5, fs*3):
        grams.append(data[i: i+fs*5])

    return grams

def load_hs(file_path):
    hs, sr = librosa.load(file_path, sr=None)
    length = len(hs)

    return hs, length, sr

def hs_segment(data_path, ans_path):
    files = open(os.path.join(data_path, 'RECORDS'), 'r').read().splitlines()

    if not os.path.exists(ans_path):
        os.makedirs(ans_path)

    with open('./model_hs/mbcnn_backbone/hs_segmentor.json', 'r') as json_file:
        config= json_file.read()
    model = model_from_json(config, custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization, 'leaky_relu': tf.nn.leaky_relu})
    model.load_weights('./model_hs/mbcnn_backbone/hs_segmentor.h5')

    en = hs_densenet_encoder()
    de = decoder()

    x_in = Input(shape=(4000, 1))
    
    z = en(x_in, training=False)
    logits = de(z, training=False)

    for k, file in enumerate(files):
        print(file)
        file_path = os.path.join(data_path, file+'.wav')
        hs, length, sr = load_hs(file_path)
        hs = hsp.butter_bandpass_filter(hs, 15, 400, fs=sr, order=5)
        hs = hsp.downsample(hs, sr, 800)

        hs_batch = ngrams(hs, len(hs), 800)
        hs_batch = np.array(hs_batch)
        hs_batch = hsp.hs_process_batch(hs_batch)

        preds = []

        logits = model(hs_batch, training=False)
        logits = np.around(logits, decimals=2)
        logits = np.reshape(logits, (len(hs_batch), 250, 4))
        preds.extend(logits)

        hs_last = hs[-800*5: ]
        hs_last = hsp.hs_process(hs_last)
        hs_last = np.expand_dims(hs_last, 0)

        logits_last = model(hs_last, training=False)
        logits_last = np.around(logits_last, decimals=2)

        gram_len = sr*3*(len(hs_batch)-2) + 2*int(sr*4)
        t_remain = (length-gram_len) / sr
        preds_remain = logits_last[0, -1*int((t_remain+1)*50):, :]

        preds = np.array(preds)
        if len(hs_batch) == 1:
            preds = preds[0, : 200, :]
        else:
            preds_init = preds[0, : 200, :]
            preds_post = preds[1: , 50: 200, :]
            preds_post = np.reshape(preds_post, (np.shape(preds_post)[0]*np.shape(preds_post)[1], -1))
            preds = np.insert(preds_post, 0, preds_init, axis=0)

        preds = np.insert(preds, -1, preds_remain, axis=0)
        preds = pp.hs_correct(preds)
        preds = np.array([[j] * 20 for j in preds]).flatten()

        np.save(os.path.join(ans_path, '{}.npy'.format(file)), preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_path',
                        help='path saving test record file')
    parser.add_argument('-r',
                        '--result_save_path',
                        help='path saving QRS-complex location results')

    args = parser.parse_args()

    hs_segment(data_path=args.data_path, ans_path=args.result_save_path)


