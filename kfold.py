
import numpy as np
import pandas as pd
import os
import pickle
import argparse
from sklearn.model_selection import KFold
from utils.data_utils import *
from collections import defaultdict

def filt_code(data, code_type, min_=5):
    """ Filter code sets based on frequency count
    Args:
        min_: (int) minimum number of occurence in order to include in final dict
        data:

    """
    codes = defaultdict(lambda : 0)
    for k, v in data.items():
        for vv in v:
            for cc in set(vv[code_type]):
                codes[cc] += 1

    keys = set(codes.keys())
    for k in keys:
        if (codes[k] < min_):
            del codes[k]
    return codes

def ret_filtered_code(codes, filt):
    return set([codes[i] for i in range(len(codes)) if codes[i] in filt])

if __name__ == '__main__':
    """Generates Kfold splits based on patient ids.
    """
    parser = argparse.ArgumentParser(description='Process Mimic-iii CSV Files')
    parser.add_argument('-p', '--path', default=None, type=str, help='path to mimic-iii csvs')
    parser.add_argument('-s', '--save', default=None, type=str, help='path to dump output')
    parser.add_argument('-seed', '--seed', default=1, type=int, help='numpy seed used to create datasplit')
    parser.add_argument('-k', '--kfold',  default=7, type=int, help='kfold split')
    parser.add_argument('-filter_codes',  action='store_true', help='filter codes based on frequency count')
    parser.add_argument('-min_adm',  type=int, help='min number of admissions filter, af must be specified')

    args = parser.parse_args()
    np.random.seed(args.seed)


    data = pickle.load(open(args.path, 'rb'))
    data_info = data['info']
    data_data = data['data']
    if (args.filter_codes):
        proc_codes = filt_code(data_data, 'procedures')
        diag_codes = filt_code(data_data, 'diagnoses')
        med_codes = filt_code(data_data, 'medications')
        for k, v in data_data.items():
            if (len(v) < args.min_adm):
                del data_data[k]
                continue
            for i in range(len(v)):
                v[i]['procedures'] = list(ret_filtered_code(v[i]['procedures'], proc_codes))
                v[i]['diagnoses'] = list(ret_filtered_code(v[i]['diagnoses'], diag_codes))
                v[i]['medications'] = list(ret_filtered_code(v[i]['medications'], med_codes))
        data_temp = {}
        data_temp['info'] = data_info
        data_temp['data'] = data_data


        try:
            with open(os.path.abspath(os.path.join(args.save, '..', 'data_filtered.pkl')), 'wb') as handle:
                pickle.dump(data, handle)
        except:
            import pdb; pdb.set_trace()

    pids = np.asarray(list(data_data.keys()))
    np.random.shuffle(pids)
    kf = KFold(args.kfold,random_state=None, shuffle=False)

    if (not os.path.isdir(args.save)):
        os.makedirs(args.save)

    for idx, ids in enumerate(kf.split(pids)):
        ids = (pids[ids[0]], pids[ids[1]])
        pickle.dump(ids, open(os.path.join(args.save, 'split_{}.pkl'.format(idx)), 'wb'))
