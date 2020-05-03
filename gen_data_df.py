import numpy as np
import pandas as pd
import os
import pickle
import argparse
from utils.data_utils import *


if __name__ == '__main__':

    """
        Generate dataframe where each row represents patient admission
    """

    parser = argparse.ArgumentParser(description='Process Mimic-iii CSV Files')
    parser.add_argument('-p', '--path', default=None, type=str, help='path to mimic-iii csvs')
    parser.add_argument('-s', '--save', default=None, type=str, help='path to dump output')
    parser.add_argument('-ft', '--filters_text', default=['Discharge summary', 'ECG', 'Pharmacy', 'Physician', 'Radiology', 'Respiratory'],  nargs='+')
    parser.add_argument('-min-adm', '--min_admission', default=1, type=int, help='minimum number of admissions for each patient')
    args = parser.parse_args()





    filters = args.filters_text
    patients = read_patients_table(args.path)

    # format date time
    df_adm = pd.read_csv(os.path.join(args.path, 'ADMISSIONS.csv'))
    df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    df_adm = df_adm.reset_index(drop = True)
    df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
    df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

    rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

    #When we filter out the "ELECTIVE",
    #we need to correct the next admit time
    #for these admissions since there might
    #be 'emergency' next admit after "ELECTIVE"
    df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
    df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
    df_adm['readmission_label'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')
    ### filter out newborn and death
    df_adm = df_adm[df_adm['ADMISSION_TYPE']!='NEWBORN']
    df_adm['DURATION'] = (df_adm['DISCHTIME']-df_adm['ADMITTIME']).dt.total_seconds()/(24*60*60)

    df_notes = pd.read_csv(os.path.join(args.path, 'NOTEEVENTS.csv'))
    df_notes = df_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])
    df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME','readmission_label','DURATION', 'DIAGNOSIS', 'MARITAL_STATUS', 'ETHNICITY', 'DISCHARGE_LOCATION']],
                            df_notes[['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT','CATEGORY']],
                            on = ['SUBJECT_ID','HADM_ID'],
                            how = 'left')

    # Adding clinical codes to dataset

    # add diagnoses
    diagnoses = read_icd_diagnoses_table(args.path)
    diagnoses = filter_diagnoses_codes(diagnoses)
    diagnoses = group_by_return_col_list(diagnoses, ['SUBJECT_ID', 'HADM_ID'], 'ICD9_CODE')

    # add cptevents
    cptevents = read_cptevents_table(args.path)
    cptevents = filter_cptevents_codes(cptevents)
    cptevents = group_by_return_col_list(cptevents, ['SUBJECT_ID', 'HADM_ID'], 'CPT_CD')

    # add prescriptions
    prescriptions = read_prescriptions_table(args.path)
    prescriptions = filter_prescription_codes(prescriptions)
    prescriptions = group_by_return_col_list(prescriptions, ['SUBJECT_ID', 'HADM_ID'], 'NDC')

    # add procedures
    procedures = read_icd_procedures_table(args.path)
    procedures = filter_procedure_codes(procedures)
    procedures = group_by_return_col_list(procedures, ['SUBJECT_ID', 'HADM_ID'], 'ICD9_CODE', 'ICD9_CODE_PROCEDURE')

    stays = read_icustays_table(args.path)

    stays = merge_on_subject(stays, patients)
    stays = merge_on_subject_admission_left(stays, diagnoses)
    stays = merge_on_subject_admission_left(stays, cptevents)
    stays = merge_on_subject_admission_left(stays, prescriptions)
    stays = merge_on_subject_admission_left(stays, procedures)
    stays = add_age_to_icustays(stays)

    df_adm_notes = pd.merge(df_adm_notes, stays, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    filt = df_adm_notes['ICD9_CODE'].isna() & df_adm_notes['CPT_CD'].isna()
    df_adm_notes = df_adm_notes[~filt]

    df_adm_notes['ADMITTIME_C'] = df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0])
    df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes.ADMITTIME_C, format = '%Y-%m-%d', errors = 'coerce')
    df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')

    filt = df_adm_notes['CATEGORY'].apply(lambda x: x in filters)
    df_adm_notes = df_adm_notes[filt]

    ### If Discharge Summary
    df_discharge = df_adm_notes[df_adm_notes['CATEGORY'] == 'Discharge summary']
    # multiple discharge summary for one admission -> after examination -> replicated summary -> replace with the last one
    df_discharge = (df_discharge.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
    df_discharge = df_discharge[df_discharge['TEXT'].notnull()]

    df_discharge = remove_min_admissions(df_discharge, min_admits=args.min_admission)


    df_adm_notes = df_adm_notes[df_adm_notes['CATEGORY'] != 'Discharge summary']

    ### If Less than n days on admission notes (Early notes)
    def less_n_days_data (df_adm_notes, n):
        df_less_n = df_adm_notes[((df_adm_notes['CHARTDATE']-df_adm_notes['ADMITTIME_C']).dt.total_seconds()/(24*60*60))<n]
        df_less_n=df_less_n[df_less_n['TEXT'].notnull()]
        return df_less_n

    df_less_1 = less_n_days_data(df_adm_notes, 1)
    df_less_2 = less_n_days_data(df_adm_notes, 2)

    import re

    def preprocess1(x):
        y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
        y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
        y=re.sub('dr\.','doctor',y)
        y=re.sub('m\.d\.','md',y)
        y=re.sub('admission date:','',y)
        y=re.sub('discharge date:','',y)
        y=re.sub('--|__|==','',y)
        return y

    def preprocessing(df_less_n):
        df_less_n['TEXT']=df_less_n['TEXT'].fillna(' ')
        df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\n',' ')
        df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\r',' ')
        df_less_n['TEXT']=df_less_n['TEXT'].apply(str.strip)
        df_less_n['TEXT']=df_less_n['TEXT'].str.lower()
        df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: preprocess1(x))
        df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: " ".join(x.split()))
        return df_less_n

    def append_text(df):
        hadm_ids = set(df['HADM_ID'])
        t_df = pd.DataFrame()
        for hid in hadm_ids:
            t = df[df['HADM_ID'] == hid]
            t = t.sort_values('ADMITTIME')
            td = t[t['CATEGORY'] == 'Discharge summary']
            tr = t[t['CATEGORY'] != 'Discharge summary']
            tr = " ".join(tr['TEXT'])
            td = " ".join(td['TEXT'])
            t = t.iloc[0]
            t['TEXT_DISCHARGE'] = td
            t['TEXT_REST'] = tr
            t_df = t_df.append(t)

        t_df['TEXT_DISCHARGE'] = t_df['TEXT_DISCHARGE'].astype(str)
        t_df['TEXT_REST'] = t_df['TEXT_REST'].astype(str)
        return t_df

    df_less_1 = df_less_1.append(df_discharge).reset_index()
    df_less_1 = preprocessing(df_less_1)
    df_less_1 = append_text(df_less_1)
    df_less_1 = compute_time_delta(df_less_1)

    df_less_2 = df_less_2.append(df_discharge).reset_index()
    df_less_2 = compute_time_delta(preprocessing(df_less_2))
    df_less_2 = append_text(df_less_2)

    if (not os.path.isdir(args.save)):
        os.makedirs(args.save)

    with open(os.path.join(args.save, 'df_all.pkl'), 'wb') as handle:
        pickle.dump(df_adm_notes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.save, 'df_less_1.pkl'), 'wb') as handle:
        pickle.dump(df_less_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.save, 'df_less_2.pkl'), 'wb') as handle:
        pickle.dump(df_less_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
