import numpy as np
import random
import os
import re


class QRSDataset():
    def __init__(self, batch_size, cv_path, data_path, label_path, fold=1):
        self.batch_size = batch_size
        self.cv_path = cv_path
        self.data_path = data_path
        self.label_path = label_path
        self.fold = fold

    def inputs(self, is_training=True):
        train_set = open(os.path.join(self.cv_path, 'train_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()
        test_set = open(os.path.join(self.cv_path, 'test_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        if is_training:
            train_data_list = random.sample(train_set, self.batch_size)
            for i, data_index in enumerate(train_data_list):
                index = re.split('[_.]', data_index)[1]
                data_file = os.path.join(self.data_path, data_index+'.npy')
                label_file = os.path.join(self.label_path, 'R_'+index+'.npy')

                ecg_sample = np.load(data_file)
                r_ref = np.load(label_file)

                train_data.append(ecg_sample)
                train_labels.append(r_ref)

            train_data = np.array(train_data)
            train_labels = np.asarray(train_labels)

            train_data = np.expand_dims(train_data, -1)

            return train_data, train_labels
        else:
            for i, data_index in enumerate(test_set):
                index = re.split('[_.]', data_index)[1]
                data_file = os.path.join(self.data_path, data_index+'.npy')
                label_file = os.path.join(self.label_path, 'R_'+index+'.npy')

                ecg_sample = np.load(data_file)
                r_ref = np.load(label_file)

                test_data.append(ecg_sample)
                test_labels.append(r_ref)

            test_data = np.array(test_data)
            test_labels = np.asarray(test_labels)

            test_data = np.expand_dims(test_data, -1)

            return test_data, test_labels

class HSDataset():
    def __init__(self, batch_size, cv_path, data_path, label_path, fold=1):
        self.batch_size = batch_size
        self.cv_path = cv_path
        self.data_path = data_path
        self.label_path = label_path
        self.fold = fold

    def convert_to_one_hot(Y, C):
        Y = np.eye(C)[Y.reshape(-1)]
        return Y

    def inputs(self, is_training=True):
        train_set = open(os.path.join(self.cv_path, 'train_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()
        test_set = open(os.path.join(self.cv_path, 'test_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        if is_training:
            train_data_list = random.sample(train_set, self.batch_size)
            for i, data_index in enumerate(train_data_list):
                index = re.split('[_.]', data_index)[1]
                data_file = os.path.join(self.data_path, data_index+'.npy')
                label_file = os.path.join(self.label_path, 'R_'+index+'.npy')

                hs_sample = np.load(data_file)
                hs_ref = np.load(label_file)
                hs_ref = convert_to_one_hot(r_ref, 4)

                train_data.append(hs_sample)
                train_labels.append(hs_ref)

            train_data = np.array(train_data)
            train_labels = np.asarray(train_labels)

            return train_data, train_labels
        else:
            for i, data_index in enumerate(test_set):
                index = re.split('[_.]', data_index)[1]
                data_file = os.path.join(self.data_path, data_index+'.npy')
                label_file = os.path.join(self.label_path, 'R_'+index+'.npy')

                hs_sample = np.load(data_file)
                hs_ref = np.load(label_file)
                hs_ref = convert_to_one_hot(hs_ref, 4)

                test_data.append(hs_sample)
                test_labels.append(hs_ref)

            test_data = np.array(test_data)
            test_labels = np.asarray(test_labels)

            return test_data, test_labels