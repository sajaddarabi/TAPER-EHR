import pandas as pd
import os
from tqdm import tqdm
from model.bert_things.pytorch_pretrained_bert.tokenization import BertTokenizer
from model.bert_things.pytorch_pretrained_bert import BertConfig, BertModel, BertPreTrainedModel
from model.bert_text_model import BertTextModel
from data_loader.utils.vocab import Vocab
import pickle
import sys
import logging
import numpy as np
import argparse
import torch
# ignore warnings from tokenization (sequence length is very long)
logging = logging.getLogger("model.bert_things.pytorch_pretrained_bert.tokenization").setLevel(logging.CRITICAL)


def embed_text(text_codes, device, bert, bert_seq_length=512, max_seq_len_text=30):
    """
        Embed text using the pretrained bert model, this is done to speed up training later (we keep the bert model fixed, you can explore
        fine tuning bert model also for downstream tasks..)
        Args:

            text_codes: tokenized text
            device: device to run model on
            bert: the bert model
            bert_seq_length: maximum bert sequence length
            max_seq_len_text: maximum occuring text sequence length in the corpus in terms of bert_seq_length

        Output:
          pooled_output: embedding
    """
    x_text = torch.zeros((bert_seq_length, max_seq_len_text), dtype=torch.long)
    x_mask = torch.zeros((bert_seq_length, max_seq_len_text,))
    n = len(text_codes) // bert_seq_length - 1
    for i in range(len(text_codes) // bert_seq_length):
        x_text[:, i] = torch.Tensor(text_codes[i * bert_seq_length: (1 + i) * bert_seq_length])
        x_mask[:, i] = 1
    if (n * bert_seq_length <= len(text_codes)):
        x_mask[len(text_codes) - bert_seq_length * (n + 1), n] = 1
        x_text[:len(text_codes) - bert_seq_length * (n + 1), n] = torch.Tensor(text_codes[(n + 1) * bert_seq_length:])
    x_text = x_text.to(device)
    x_mask = x_mask.to(device)

    with torch.no_grad():
        _, pooled_output = bert(x_text.t(), attention_mask=x_mask.t())
    return pooled_output

def compute_max_seq_len_text(df, col, tokenizer):
    """
        Compute the maximum occuring sequence length in the dataset
        Args:

            df: dataframe containing the complete dataset
            col: column name containing the text
            tokenizer: map used to convert tokents to ids (assumes tokenier has a convert_tokens_to_ids function)

        Output:
           max_seq_len_text: (int)
    """
    max_seq_len_text = 0
    for i, r in df.iterrows():
        text = r[col]
        ttok = tokenizer.tokenize(text)
        ttok = tokenizer.convert_tokens_to_ids(ttok)
        if (len(ttok) > max_seq_len_text):
            max_seq_len_text = len(ttok)
    return max_seq_len_text

def _prepare_device(n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        if n_gpu_use = 0, use cpu
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            logging.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = -1
        if n_gpu_use > n_gpu:
            logging.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

def main():
    """

    Will generate a dictionary as follows:
        <key> patientid : <value> lsit of dicts, where each dict contains admission data
                                  [
                                  {<key> feature/label name : <value> feature/label value}
                                  ]

    """
    parser = argparse.ArgumentParser(description='Generate Text+Code dataset')
    parser.add_argument('-p', '--path', default=None, type=str, help='path to pandas dataframe where rows are admissions')
    parser.add_argument('-vp', '--vocab_path', default='', type=str, help='path to where code vocabulary are stored assumes diagnoses vocab file named as diag.vocab and cpt vocab as cpt.vocab')
    parser.add_argument('-s', '--save', default='./', type=str, help='path to save pkl files')
    parser.add_argument('-et', '--embed_text', default=False, action='store_true', help='flag wether to embed text or not')
    parser.add_argument('-cpb', '--bert_config_path', default=None, type=str, help='path to bert config')
    parser.add_argument('-vpb', '--bert_vocab_path', default=None, type=str, help='path to bert vocab ')
    parser.add_argument('-sdp', '--state_dict_path', default=None, type=str, help='path to bert state dict')
    parser.add_argument('-gpu', '--gpu', default=0, type=int)
    parser.add_argument('-bsl', '--max_bert_seq_len', default=512, type=int, help='maximum sequence length of bert model')
    parser.add_argument('-tsld', '--text_seq_length_discharge', default=0, type=int, help='pass this if maximum text sequence length is known for discharge text to avoid long processing time')
    parser.add_argument('-tslr', '--text_seq_length_rest', default=0, type=int, help='pass this if maximum text sequence length is known for rest of text (other than discharge) to avoid longer processing time')
    parser.add_argument('-sc', '--short_code', default=False, action='store_true', help='flag for using short codes ')
    parser.add_argument('-diag', '--diagnoses', default=False, action='store_true', help='flag for including diagnoses codes')
    parser.add_argument('-proc', '--procedures', default=False, action='store_true', help='flag for including procedures codes')
    parser.add_argument('-med', '--medications', default=False, action='store_true', help='flag for including medication codes')
    parser.add_argument('-cpt', '--cpts', default=False, action='store_true', help='flag for including cpt codes')

    parser.add_argument('-ma', '--min_adm', default=0, type=int)

    args = parser.parse_args()
    df = pd.read_pickle(args.path)
    df_orig = df
    # remove organ donor admissions
    if ('DIAGNOSIS' in df.columns):

        REMOVE_DIAGNOSIS = ~((df['DIAGNOSIS'] == 'ORGAN DONOR ACCOUNT') | (df['DIAGNOSIS'] == 'ORGAN DONOR') | \
                       (df['DIAGNOSIS'] == 'DONOR ACCOUNT'))
        df = df[REMOVE_DIAGNOSIS]

    df = df[~df['ICD9_CODE'].isna()] # drop patients with no icd9 code?
    df = df[~(df['TEXT_REST'].isna() | df['TEXT_REST'].isna())]

    if ('TIMEDELTA' in df.columns):
        df['TIMEDELTA'] = df['TIMEDELTA'].fillna(pd.to_timedelta("0"))
        df['TIMEDELTA'] = pd.to_timedelta(df['TIMEDELTA'])
        df['TIMEDELTA'] = df['TIMEDELTA'].apply(lambda x: x.seconds)

    pids = list(set(df['SUBJECT_ID'].tolist()))

    # lambda
    demographic_cols = {'AGE': [], 'GENDER': [], 'LAST_CAREUNIT': [],
                        'MARITAL_STATUS': [], 'ETHNICITY': [],
                        'DISCHARGE_LOCATION': []}

    df.loc[:, 'MARITAL_STATUS'], demographic_cols['MARITAL_STATUS'] = pd.factorize(df['MARITAL_STATUS'])
    df.loc[:, 'ETHNICITY'], demographic_cols['ETHNICITY'] = pd.factorize(df['ETHNICITY'])
    df.loc[:, 'DISCHARGE_LOCATION'], demographic_cols['DISCHARGE_LOCATION'] = pd.factorize(df['DISCHARGE_LOCATION'])
    df.loc[:, 'LAST_CAREUNIT'], demographic_cols['LAST_CAREUNIT'] = pd.factorize(df['LAST_CAREUNIT'])
    df.loc[:, 'GENDER'], demographic_cols['GENDER'] = pd.factorize(df['GENDER'])
    df.loc[:, 'AGE'] = df['AGE'].astype(int)
    los_bins = [1, 2, 3, 4, 5, 6, 7, 8, 14, float('inf')]
    los_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df.loc[:, 'LOS'] = pd.cut(df['LOS'], bins=los_bins, labels=los_labels)



    temp_data = []
    data = {}

    diag_vocab = Vocab()
    cpt_vocab = Vocab()
    med_vocab = Vocab()
    proc_vocab = Vocab()


    if (args.vocab_path != ''):
        #to use below checkout https://github.com/sajaddarabi/HCUP-US-EHR
        if (args.diagnoses):
            diag_vocab._build_from_file(os.path.join(args.vocab_path, 'diag.vocab'))
        if (args.cpts):
            cpt_vocab._build_from_file(os.path.join(args.vocab_path, 'cpt.vocab'))
        #if (args.procedures):
        #    proc_vocab._build_from_file(os.path.join(args.vocab_path, 'proc.vocab'))
        #if (args.med):
            #med_vocab._build_from_file(os.path.join(args.vocab_path, 'med.vocab'))



    if (os.path.exists(os.path.join(args.save, 'data.pkl'))):
        temp_data = pickle.load(open(os.path.join(args.save, 'data.pkl'), 'rb'))
        temp_data = temp_data['data']

        t = list(temp_data.keys())
        t = t[0]
        d =  'text_embedding' in temp_data[t][0]

        if (not d):
            temp_data = []
        else:
            model = None
            bert_config = None
            torch.cuda.empty_cache()

    if args.embed_text:
        tokenizer = BertTokenizer(args.bert_vocab_path)

    if args.embed_text and (len(temp_data) == 0):
        bert_config = BertConfig(args.bert_config_path)
        model = BertTextModel(bert_config)
        state_dict = torch.load(args.state_dict_path)
        model.init_bert_weights(state_dict)
        device, _ = _prepare_device(args.gpu)
        model = model.to(device)
        max_seq_len_text_d = args.text_seq_length_discharge
        max_seq_len_text_r = args.text_seq_length_rest

        if max_seq_len_text_d  == 0:
            max_seq_len_text = compute_max_seq_len_text(df, 'TEXT_DISCHARGE', tokenizer)
            max_seq_len_text = max_seq_len_text // args.max_bert_seq_len + 1
            max_seq_len_text_d = max_seq_len_text
            print("text sequence discharge length: {}".format(max_seq_len_text_d))

        if max_seq_len_text_r  == 0:
            max_seq_len_text = compute_max_seq_len_text(df, 'TEXT_REST', tokenizer)
            max_seq_len_text = max_seq_len_text // args.max_bert_seq_len + 1
            max_seq_len_text_r = max_seq_len_text
            print("text sequence rest length: {}".format(max_seq_len_text_r))
    try:
        for pid in tqdm(pids):
            pid_df = df[df['SUBJECT_ID'] == pid]
            pid_df = pid_df.sort_values('ADMITTIME').reset_index()
            if (len(pid_df) < 1): # must atleast have two data points
                continue
            data[pid] = []

            t = 0
            hadm_ids = set(df['HADM_ID'])
            for i, r in pid_df.iterrows():
                #filt notes prior to n days and concatenate them
                # leave discharge summary seperate
                admit_data = {}
                demographics = [r['AGE'], r['GENDER'], r['MARITAL_STATUS']]

                icu_unit = np.zeros((demographic_cols['LAST_CAREUNIT'].size, ), dtype=int)
                icu_unit[r['LAST_CAREUNIT']] = 1
                demographics += list(icu_unit)

                ethnicity = np.zeros((demographic_cols['ETHNICITY'].size, ), dtype=int)
                ethnicity[r['ETHNICITY']] = 1
                demographics += list(ethnicity)

                ethnicity = np.zeros((demographic_cols['ETHNICITY'].size, ), dtype=int)
                ethnicity[r['ETHNICITY']] = 1
                demographics += list(ethnicity)

                admit_data['demographics'] = demographics
                dtok, ptok, mtok, ctok = [], [], [], []
                diagnosis_codes, proc_codes, med_codes, cpt_codes = np.nan, np.nan, np.nan, np.nan

                if args.diagnoses:
                    diagnosis_codes = r['ICD9_CODE']

                if (diagnosis_codes == diagnosis_codes):
                    dtok = diag_vocab.convert_to_ids(diagnosis_codes , 'D', args.short_code)

                if (args.procedures):
                    proc_codes = r['ICD9_CODE_PROCEDURE']

                if (proc_codes == proc_codes):
                    ptok = proc_vocab.convert_to_ids(proc_codes, 'P', args.short_code)

                if args.medications:
                    med_codes = r['NDC'] # issue with NDC what mapping version is being used..?

                if (med_codes == med_codes):
                    mtok = med_vocab.convert_to_ids(med_codes, 'M')

                if args.cpts:
                    cpt_codes = r['CPT_CD']

                if (cpt_codes == cpt_codes):
                    ctok = cpt_vocab.convert_to_ids(cpt_codes, 'C')

                admit_data['diagnoses']  = dtok
                admit_data['procedures'] = ptok
                admit_data['medications'] = mtok
                admit_data['cptproc'] = ctok

                if (r['TIMEDELTA'] == r['TIMEDELTA']):
                    t += r['TIMEDELTA']

                admit_data['timedelta'] = t

                text_discharge = r['TEXT_DISCHARGE']
                text_rest = r['TEXT_REST']

                ttokd = tokenizer.tokenize(text_discharge)
                ttokd = tokenizer.convert_tokens_to_ids(ttokd)
                ttokr = tokenizer.tokenize(text_rest)
                ttokr = tokenizer.convert_tokens_to_ids(ttokr)

                admit_data['text_discharge_raw'] = text_discharge
                admit_data['text_rest_raw'] = text_rest

                admit_data['text_discharge_len'] = len(ttokd)
                admit_data['text_rest_len'] = len(ttokr)

                admit_data['text_discharge_token'] = ttokd
                admit_data['text_rest_token'] = ttokr

                if len(temp_data) == 0:
                    if (args.embed_text):
                        ttokd = embed_text(ttokd, device, model, args.max_bert_seq_len, max_seq_len_text_d)
                        ttokd = ttokd.cpu().numpy()
                        ttokr = embed_text(ttokr, device, model, args.max_bert_seq_len, max_seq_len_text_r)
                        ttokr = ttokr.cpu().numpy()
                else:
                    ttok = temp_data[pid][i]['text_embedding']

                admit_data['text_embedding_discharge'] = ttokd
                admit_data['text_embedding_rest'] = ttokr

                admit_data['los'] = r['LOS']
                admit_data['readmission'] = r['readmission_label']
                admit_data['mortality'] = r['DEATHTIME'] == r['DEATHTIME']
                data[pid].append(admit_data)

    except Exception as error:
        print(error)
        import pdb; pdb.set_trace()

    if (not os.path.exists(args.save)):
        os.makedirs(args.save)


    # temporarly save data incase something goes wrong ...
    try:
        with open(os.path.join(args.save, 'data.pkl'), 'wb') as handle:
            data_dict = {}
            data_dict['data'] = data
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        import pdb; pdb.set_trace()

    pids = list(data.keys())
    flatten = lambda x: [item for sublist in x for item in sublist]

    data_info = {}
    num_icd9_codes, num_proc_codes, num_med_codes = 0, 0, 0
    data_info['num_patients'] = len(pids)
    data_info['max_seq_len_text_d'] = max_seq_len_text_d
    data_info['max_seq_len_text_r'] = max_seq_len_text_r

    data_info['num_icd9_codes'] = 0
    data_info['num_proc_codes'] = 0
    data_info['num_med_codes'] = 0

    if (args.diagnoses):
        num_icd9_codes = len(set(flatten(df_orig['ICD9_CODE'].dropna())))

    data_info['num_icd9_codes'] = num_icd9_codes

    if (args.procedures):
        num_proc_codes = len(set(flatten(df_orig['ICD9_CODE_PROCEDURE'].dropna())))

    data_info['num_proc_codes'] = num_proc_codes

    if (args.medications):
        num_med_codes = len(set(flatten(df_orig['NDC'].dropna())))

    data_info['num_med_codes'] = num_med_codes
    data_info['demographics_shape'] = len(data[pids[0]][0]['demographics'])
    data_info['demographic_cols'] = demographic_cols
    data_info['total_codes'] = data_info['num_icd9_codes'] + data_info['num_proc_codes'] + data_info['num_med_codes']

    if (not os.path.exists(args.save)):
        os.makedirs(args.save)

    with open(os.path.join(args.save, 'data.pkl'), 'wb') as handle:
        data_dict = {}
        data_dict['info'] = data_info
        data_dict['data'] = data
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.save, 'cpt_vocab.pkl'), 'wb') as handle:
        pickle.dump(cpt_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save, 'diag_vocab.pkl'), 'wb') as handle:
        pickle.dump(diag_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save, 'med_vocab.pkl'), 'wb') as handle:
        pickle.dump(med_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save, 'proc_vocab.pkl'), 'wb') as handle:
        pickle.dump(proc_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
