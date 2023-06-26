import argparse
import math
import numpy as np
import os

import pandas as pd


STATE = {'s1', 'sys', 's2', 'dia'}

def load_label(label_path):
    df = pd.read_csv(label_path, names=['START', 'STATES'])
    starts = np.asarray(df['START']) * 2000
    starts = starts.astype(int)
    states = np.asarray(df['STATES'], dtype=str)

    s1 = []
    s2 = []
    sys = []
    dia = []
    noise = []
    dict = {}
    for (start, state) in zip(starts, states):
        if state == 'S1':
            s1.append(str(start))
        elif state == 'S2':
            s2.append(str(start))
        elif state == 'systole':
            sys.append(str(start))
        elif state == 'diastole':
            dia.append(str(start))
        elif state == '(N' or state == 'N)':
            noise.append(str(start))

    dict = {'s1': s1, 's2': s2, 'systole': sys, 'diastole': dia, 'noise': noise}

    return dict

def load_pred(pred_path):
    '''
    preds label:
    s1: 0; systole: 1; s2: 2; diastole: 3; noise: 4
    start_location when:
    sys_start: diff = 1 & preds[diff_index] = 0 -> index+1
    s2_start: diff = 1 & preds[diff_index] = 1
    dia_start: diff = 1 & preds[diff_index] = 2
    s1_start: diff = -3 &  preds[diff_index] = 3
    '''
    s1 = []
    sys = []
    s2 = []
    dia = []
    noise = []
    dict = {}

    preds = np.load(pred_path)

    diff = np.diff(preds)
    starts = np.where(np.logical_or(diff == 1, diff == -3))[0]
    for location in starts:
        if preds[location] == 3 and diff[location] == -3:
            s1.append(int((location + 1)*2))
        elif preds[location] == 0 and diff[location] == 1:
            sys.append(int((location + 1)*2))
        elif preds[location] == 1 and diff[location] == 1:
            s2.append(int((location + 1)*2))
        elif preds[location] == 2 and diff[location] == 1:
            dia.append(int((location + 1)*2))

    dict = {'s1': s1, 's2': s2, 'systole': sys, 'diastole': dia}

    return dict

def count_pn(ans, ann, noise, thr_, fs):
    tp = 0
    fn = 0
    fp = 0
    ans = np.array(ans)
    ann = np.array(ann)
    noise = np.array(noise)

    if len(noise) > 0:
        noise_count = len(noise) // 2
        for i in range(noise_count):
            ans = ans[(ans < noise[i*2]) | (ans > noise[i*2+1])]

    for i in range(len(ann)):
        ss1 = ann[i] - thr_ if ann[i] >= thr_ else 0
        ee = ann[i] + thr_
        ss2 = ann[i+1] - thr_ if i < len(ann) - 1 else (ee + fs / 4)

        cur_index = list(np.where((ans >= ss1) & (ans <= ee))[0])
        err_index = list(np.where((ans > ee) & (ans < ss2))[0])

        if len(cur_index) != 0:
            if len(cur_index) == 1:
                tp = tp + 1
            else:
                tp = tp + 1
                fp = fp + len(cur_index) - 1
        else:
            fn = fn + 1

        if len(err_index) != 0:
            fp = fp + len(err_index)

    return tp, fp, fn

def Standard_error(sample):

    std=np.std(sample, ddof=0)

    standard_error = std / math.sqrt(len(sample))

    return standard_error

def score_std(answer_path, annotation_path, thr_, fs_):
    '''
    20ms-100ms precision
    '''
    def is_npy(f):
        return f.endswith('.npy')

    NUM_BEAT = 0
    answers = list(filter(is_npy, os.listdir(answer_path)))

    F1_state = np.zeros((len(answers), 4))
    f1_total = []
    se_total = []
    sp_total = []
    count = 0
    for answer in answers:
        TP_all = 0
        FN_all = 0
        FP_all = 0
        index = answer.split('.')[0]
        if not os.path.exists(os.path.join(annotation_path, index+'.csv')):
            continue
        ans = load_pred(os.path.join(answer_path, answer))
        ann = load_label(os.path.join(annotation_path, index+'.csv'))

        ans_s1 = [int(l) for l in ans['s1']]
        ans_sys = [int(l) for l in ans['systole']]
        ans_s2 = [int(l) for l in ans['s2']]
        ans_dia = [int(l) for l in ans['diastole']]

        ann_s1 = [int(l) for l in ann['s1']]
        ann_sys = [int(l) for l in ann['systole']]
        ann_s2 = [int(l) for l in ann['s2']]
        ann_dia = [int(l) for l in ann['diastole']]
        ann_noise = [int(l) for l in ann['noise']]

        ans_all = [ans_s1, ans_sys, ans_s2, ans_dia]
        ann_all = [ann_s1, ann_sys, ann_s2, ann_dia]

        for i in range(4):
            statics = {'RECORD': [], 'TP': [], 'FP': [], 'FN': []}
            tp, fp, fn = count_pn(ans_all[i], ann_all[i], ann_noise, thr_, fs_)
            se = (tp+0.001) / (tp + fn +0.001) * 100
            sp = (tp+0.001) / (tp + fp + 0.001) * 100

            f1 = 2 * se * sp / (se + sp)
            F1_state[count][i] = f1

            TP_all = TP_all + tp
            FP_all = FP_all + fp
            FN_all = FN_all + fn
            #print(index + ': TP: {}; FP: {}; FN: {}'.format(TP_all, FP_all, FN_all))

        #acc = TP_all / (TP_all + FP_all + FN_all) * 100
        se = (TP_all+0.001) / (TP_all + FN_all) * 100
        sp = (TP_all+0.001) / (TP_all + FP_all) * 100
        f1 = 2 * se * sp / (se + sp)

        se_total.append(se)
        sp_total.append(sp)
        f1_total.append(f1)

        count += 1

    se_total = np.array(se_total)
    sp_total = np.array(sp_total)
    f1_total = np.array(f1_total)

    s1_f1_mean = np.mean(F1_state[:, 0])
    sys_f1_mean = np.mean(F1_state[:, 1])
    s2_f1_mean = np.mean(F1_state[:, 2])
    dia_f1_mean = np.mean(F1_state[:, 3])
    s1_f1_std = Standard_error(F1_state[:, 0])
    sys_f1_std = Standard_error(F1_state[:, 1])
    s2_f1_std = Standard_error(F1_state[:, 2])
    dia_f1_std = Standard_error(F1_state[:, 3])

    se_mean = np.mean(se_total)
    se_std = Standard_error(se_total)
    sp_mean = np.mean(sp_total)
    sp_std = Standard_error(sp_total)
    f1_mean = np.mean(f1_total)
    f1_std = Standard_error(f1_total)

    print('F1_S1: {}; STD_S1: {}'.format(s1_f1_mean, s1_f1_std))
    print('F1_SYS: {}; STD_SYS: {}'.format(sys_f1_mean, sys_f1_std))
    print('F1_S2: {}; STD_S2: {}'.format(s2_f1_mean, s2_f1_std))
    print('F1_DIA: {}; STD_DIA: {}'.format(dia_f1_mean, dia_f1_std))
    print('SE: {}; STD: {}'.format(se_mean, se_std))
    print('SP: {}; STD: {}'.format(sp_mean, sp_std))
    print('F1_total: {}; STD: {}'.format(f1_mean, f1_std))

def score(answer_path, annotation_path, thr_, fs_):
    '''
    20ms-100ms precision
    '''
    def is_npy(f):
        return f.endswith('.npy')

    NUM_BEAT = 0
    answers = list(filter(is_npy, os.listdir(answer_path)))

    TP_total = 0
    FN_total = 0
    FP_total = 0
    for i in range(4):
        statics = {'RECORD': [], 'TP': [], 'FP': [], 'FN': []}
        TP_all = 0
        FN_all = 0
        FP_all = 0
        for answer in answers:
            TP = 0
            FN = 0
            FP = 0
            index = answer.split('.')[0]
            if not os.path.exists(os.path.join(annotation_path, index+'.csv')):
                continue
            ans = load_pred(os.path.join(answer_path, answer))
            ann = load_label(os.path.join(annotation_path, index+'.csv'))

            ans_s1 = [int(l) for l in ans['s1']]
            ans_sys = [int(l) for l in ans['systole']]
            ans_s2 = [int(l) for l in ans['s2']]
            ans_dia = [int(l) for l in ans['diastole']]

            ann_s1 = [int(l) for l in ann['s1']]
            ann_sys = [int(l) for l in ann['systole']]
            ann_s2 = [int(l) for l in ann['s2']]
            ann_dia = [int(l) for l in ann['diastole']]
            ann_noise = [int(l) for l in ann['noise']]

            ans_all = [ans_s1, ans_sys, ans_s2, ans_dia]
            ann_all = [ann_s1, ann_sys, ann_s2, ann_dia]
            '''
            tp, fp, fn = count_pn(ans_dia, ann_dia, ann_noise, thr_, fs_)
            TP = TP + tp
            FP = FP + fp
            FN = FN + fn

            '''

            tp, fp, fn = count_pn(ans_all[i], ann_all[i], ann_noise, thr_, fs_)
            TP = TP + tp
            FP = FP + fp
            FN = FN + fn

            statics['RECORD'].append(index)
            statics['TP'].append(TP)
            statics['FP'].append(FP)
            statics['FN'].append(FN)

            TP_all = TP_all + tp
            FP_all = FP_all + fp
            FN_all = FN_all + fn

            # print(index + ': TP: {}; FP: {}; FN: {}'.format(TP, FP, FN))
        df = pd.DataFrame(statics)
        df.to_csv(os.path.join(answer_path, 'count_{}.csv'.format(i)), index=None)

        #acc = TP_all / (TP_all + FP_all + FN_all) * 100
        se = TP_all / (TP_all + FN_all) * 100
        sp = TP_all / (TP_all + FP_all) * 100
        f1 = 2 * se * sp / (se + sp)
        print(str(i)+': ')
        #print('ACC: {}'.format(acc))
        print('F1: {}'.format(f1))

        TP_total += TP_all
        FN_total += FN_all
        FP_total += FP_all
    #acc = TP_total / (TP_total + FP_total + FN_total)
    SE = TP_total / (TP_total + FN_total) * 100
    SP = TP_total / (TP_total + FP_total) * 100
    F1 = 2 * SE * SP / (SE + SP)
    #print('ACC: {}'.format(acc))
    print('F1_total: {}'.format(F1))
    print('SE: {}'.format(SE))
    print('SP: {}'.format(SP))


if __name__ == '__main__':
    THRESHOLD = 200
    SR = 2000
    parser = argparse.ArgumentParser()
    parser.add_argument('-a',
                        '--anntation_path',
                        help='path saving test anntation file')
    parser.add_argument('-r',
                        '--result_save_path',
                        help='path saving QRS-complex location results')

    args = parser.parse_args()

    score(args.result_save_path, args.anntation_path, THRESHOLD, SR)
