import argparse
import numpy as np
import os
import re

import scipy.io as sio
import wfdb


def load_ref(database, file_path, lead=None):
    if database == 'cpsc2019':
        fs = 500
        ref = sio.loadmat(file_path)['R_peak'].squeeze()
        beats = ref[(ref >= 0.5*fs) & (ref <= 9.5*fs)]
        lead_num = 1
    elif database == 'ludb':
        sig, fields = wfdb.rdsamp(file_path)
        fs = fields['fs']
        real_r = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
        ann_ref = wfdb.rdann(file_path, 'atr_'+lead)
        beats = ann_ref.sample[np.isin(ann_ref.symbol, real_r)]
        lead_num = 12
    else:
        sig, fields = wfdb.rdsamp(file_path)
        fs = fields['fs']
        real_r = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
        ann_ref = wfdb.rdann(file_path, 'atr')
        beats = ann_ref.sample[np.isin(ann_ref.symbol, real_r)]
        lead_num = np.shape(sig)[-1]

    return fs, beats, lead_num


def score_single(database, r_ref, r_ans, fs_, thr_):
    FN = 0
    FP = 0
    TP = 0
    r_ref = np.sort(r_ref)
    if database == 'cpsc2019':
        if len(r_ref) == 0:
            FP += len(r_ans)
            TP += 0
            FN += 0
            f1 = None
        else:
            for j in range(len(r_ref)):
                loc = np.where(np.abs(r_ans - r_ref[j]) <= thr_*fs_)[0]
                if j == 0:
                    err = np.where((r_ans >= 0.5*fs_ + thr_*fs_) & (r_ans <= r_ref[j] - thr_*fs_))[0]
                elif j == len(r_ref)-1:
                    err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= 9.5*fs_ - thr_*fs_))[0]
                else:
                    err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= r_ref[j+1]-thr_*fs_))[0]

                FP = FP + len(err)
                if len(loc) >= 1:
                    TP += 1
                    FP = FP + len(loc) - 1
                elif len(loc) == 0:
                    FN += 1

            if TP == 0:
                f1 = 0
            else:
                se = TP / (TP + FN) * 100
                sp = TP / (TP + FP) * 100
                f1 = 2 * se * sp / (se + sp)

    elif database == 'ludb':
        r_ref = r_ref[(r_ref >= 1*fs_) & (r_ref <= 9*fs_)]
        r_ans = r_ans[(r_ans >= 1*fs_) & (r_ans <= 9*fs_)]
        for j in range(len(r_ref)):
            loc = np.where(np.abs(r_ans - r_ref[j]) <= thr_*fs_)[0]
            if j == 0:
                err = np.where((r_ans >= 0.5*fs_ + thr_*fs_) & (r_ans <= r_ref[j] - thr_*fs_))[0]
            elif j == len(r_ref)-1:
                err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= 9.5*fs_ - thr_*fs_))[0]
            else:
                err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= r_ref[j+1]-thr_*fs_))[0]

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN += 1

        if TP == 0:
            f1 = 0
        else:
            se = TP / (TP + FN) * 100
            sp = TP / (TP + FP) * 100
            f1 = 2 * se * sp / (se + sp)

    else:
        r_ref = r_ref[r_ref >= 1*fs_]
        r_ans = r_ans[r_ans >= 1*fs_]
        for i in range(len(r_ref)):
            loc = np.where(np.abs(r_ans - r_ref[i]) <= thr_*fs_)[0]
            if i < len(r_ref)-1:
                err = np.where((r_ans >= r_ref[i]+thr_*fs_) & (r_ans <= r_ref[i+1]-thr_*fs_))[0]

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN += 1

        if TP == 0:
            f1 = 0
        else:
            se = TP / (TP + FN) * 100
            sp = TP / (TP + FP) * 100
            f1 = 2 * se * sp / (se + sp)

    # print('F1: {}'.format(f1))
    # print('FP: {}'.format(FP))
    # print('FN: {}'.format(FN))

    return TP, FP, FN, f1

def evaluate(database, ref_path, ans_path, thr_):
    TP_all = 0
    FN_all = 0
    FP_all = 0
    files = open(os.path.join(ref_path, 'RECORDS'), 'r').read().splitlines()

    err_list = []
    for k, file in enumerate(files):
        # print(file)

        if database == 'cpsc2019':
            index = re.split('[_]', file)[-1]
            ref_file = os.path.join(ref_path, 'R_'+index+'.mat')
            fs, r_ref, lead_num = load_ref(database, ref_file)
            ans_file = os.path.join(ans_path, file+'.npy')
            r_ans = np.load(ans_file)

            tp, fp, fn, f1 = score_single(database, r_ref, r_ans, fs, thr_)

            if f1 < 80:
                err_list.append(index)
           
            TP_all += tp
            FP_all += fp
            FN_all += fn

        elif database == 'ludb':
            ref_file = os.path.join(ref_path, file)
            LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
            for j, lead in enumerate(LEADS):
                fs, r_ref, lead_num = load_ref(database, ref_file, lead)
                ans_file = os.path.join(ans_path, file+'_'+str(j)+'.npy')
                r_ans = np.load(ans_file)
                r_ans = r_ans[(r_ans >= 1*fs) & (r_ans <= 9*fs)]

                tp, fp, fn, f1 = score_single(database, r_ref, r_ans, fs, thr_)
                TP_all += tp
                FP_all += fp
                FN_all += fn
        else:
            ref_file = os.path.join(ref_path, file)
            fs, r_ref, lead_num = load_ref(database, ref_file)
            if file == 'I02':
                for i in range(11):
                    ans_file = os.path.join(ans_path, file+'_'+str(i)+'.npy')
                    r_ans = np.load(ans_file)

                    tp, fp, fn, f1 = score_single(database, r_ref, r_ans, fs, thr_)
                    TP_all += tp
                    FP_all += fp
                    FN_all += fn
            elif file == 'I58':
                for i in range(lead_num):
                    if i == 9:
                        continue
                    ans_file = os.path.join(ans_path, file+'_'+str(i)+'.npy')
                    r_ans = np.load(ans_file)

                    tp, fp, fn, f1 = score_single(database, r_ref, r_ans, fs, thr_)
                    TP_all += tp
                    FP_all += fp
                    FN_all += fn
            else:
                for i in range(lead_num):
                    ans_file = os.path.join(ans_path, file+'_'+str(i)+'.npy')
                    r_ans = np.load(ans_file)
                    if file == '207':
                        # continue
                        r_ans = r_ans[~((r_ans>14894-180)*(r_ans<21608+180))]
                        r_ans = r_ans[~((r_ans>87273-180)*(r_ans<100956+180))]
                        r_ans = r_ans[~((r_ans>554826-180)*(r_ans<589660+180))]

                    tp, fp, fn, f1 = score_single(database, r_ref, r_ans, fs, thr_)
                    TP_all += tp
                    FP_all += fp
                    FN_all += fn

    se = TP_all / (TP_all + FN_all) * 100
    sp = TP_all / (TP_all + FP_all) * 100
    f1 = 2 * se * sp / (se + sp)

    print(err_list)

    return TP_all, FP_all, FN_all, f1, se, sp


if __name__ == '__main__':
    THRESHOLD = 0.15
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--database_type',
                        help='database type')
    parser.add_argument('-a',
                        '--anntation_path',
                        help='path saving test anntation file')
    parser.add_argument('-r',
                        '--result_save_path',
                        help='path saving QRS-complex location results')

    args = parser.parse_args()

    metrics = evaluate(args.database_type, args.anntation_path, args.result_save_path, THRESHOLD)

    print('TP: {}'.format(metrics[0]))
    print('FP: {}'.format(metrics[1]))
    print('FN: {}'.format(metrics[2]))
    print('F1: {}'.format(metrics[3]))
    print('SE: {}'.format(metrics[4]))
    print('SP: {}'.format(metrics[5]))
    print('Scoring complete.')

    with open(os.path.join(args.result_save_path, 'score.txt'), 'w') as score_file:
        print('TP: {}'.format(metrics[0]), file=score_file)
        print('FP: {}'.format(metrics[1]), file=score_file)
        print('FN: {}'.format(metrics[2]), file=score_file)
        print('F1: {}'.format(metrics[3]), file=score_file)
        print('SE: {}'.format(metrics[4]), file=score_file)
        print('SP: {}'.format(metrics[5]), file=score_file)

        score_file.close()
