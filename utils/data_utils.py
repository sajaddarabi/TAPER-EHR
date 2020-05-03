from datetime import datetime
import numpy as np
import math
import pickle
import pandas as pd
import os
import sys
import re

def read_patients_table(path):
    p = pd.read_csv(os.path.join(path, 'PATIENTS.csv'))
    p = p[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD',]]
    p['DOB'] = pd.to_datetime(p['DOB'])
    p['DOD'] = pd.to_datetime(p['DOD'])
    return p

def read_cptevents_table(path):
    cpt = pd.read_csv(os.path.join(path, 'CPTEVENTS.csv'))
    cpt = cpt[['SUBJECT_ID', 'HADM_ID', 'CPT_CD',]]
    return cpt


def read_icd_procedures_table(path):
    codes = pd.read_csv(os.path.join(path, 'D_ICD_PROCEDURES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    procedures = pd.read_csv(os.path.join(path, 'PROCEDURES_ICD.csv'))
    procedures = procedures.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    procedures[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = procedures[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return procedures

def compute_time_delta(df):
    df['TIMEDELTA'] = df.sort_values(['SUBJECT_ID', 'ADMITTIME']).groupby(['SUBJECT_ID'])['ADMITTIME'].diff()

    return df

def read_admissions_table(path):
    admits = pd.read_csv(os.path.join(path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DIAGNOSIS', 'MARITAL_STATUS', 'ETHNICITY', 'DISCHARGE_LOCATION', 'ADMISSION_TYPE']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    return admits

def add_readmission_column(df_adm):
    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    df_adm = df_adm.reset_index(drop = True)
    df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
    df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID')['ADMISSION_TYPE'].shift(-1)
    rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

    df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
    df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
    return df_adm

def read_prescriptions_table(path):
    prescription = pd.read_csv(os.path.join(path, 'PRESCRIPTIONS.csv'))
    prescription = prescription[~ prescription['NDC'].isna()]
    prescription = prescription[['SUBJECT_ID', 'HADM_ID', 'NDC']].astype(int)
    prescription = prescription.dropna()
    return prescription

def filter_notes_table(admits, notes, filters = {
     'Discharge summary',
      'ECG',
      'Physician',
      'Radiology',
      'Respiratory'
      }, t=float('inf')):

    filt = notes['CATEGORY'].apply(lambda x: x in filters)
    notes = notes[filt]
    notes.loc[:, 'CHARTTIME'] = notes['CHARTTIME'].fillna(notes['CHARTDATE'])
    notes.loc[:, 'CHARTTIME'] = pd.to_datetime(notes['CHARTTIME'])
    a = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
    a.loc[:, 'ADMITTIME'] = pd.to_datetime(a['ADMITTIME'])

    tn = notes.merge(a, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
    tn['timedelta'] = (tn['CHARTTIME'] - tn['ADMITTIME']) / pd.Timedelta('1 hour')
    tn = tn[tn['timedelta'] < t]
    tn = tn.drop(columns=['ADMITTIME', 'timedelta'])
    tn = tn[tn['TEXT'].not_null()]
    return tn

def read_notes_table(path):
    notes = pd.read_csv(os.path.join(path, 'NOTEEVENTS.csv'))
    notes['TEXT'] = notes['TEXT'].fillna(' ')
    notes['TEXT'] = notes['TEXT'].str.replace('\n',' ')
    notes['TEXT'] = notes['TEXT'].str.replace('\r',' ')
    notes['TEXT'] = notes['TEXT'].apply(str.strip)
    notes['TEXT'] = notes['TEXT'].str.lower()
    notes = notes[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'TEXT']]
    return notes

def read_icustays_table(path):
    icu = pd.read_csv(os.path.join(path, 'ICUSTAYS.csv'))
    icu['INTIME'] = pd.to_datetime(icu['INTIME'])
    icu['OUTTIME'] = pd.to_datetime(icu['OUTTIME'])
    return icu

def read_icd_diagnoses_table(path):
    codes = pd.read_csv(os.path.join(path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    diagnoses = pd.read_csv(os.path.join(path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses

def filter_cptevents_codes(cpt, min_=5, max_=np.inf):
    t = cpt.groupby('CPT_CD').CPT_CD.transform(len) >= min_
    num_codes = len(set(cpt['CPT_CD']))
    num_codes_after =len(set(cpt.loc[t, 'CPT_CD']))
    print('removing cpt procedure codes occuring less than {} times. \n num codes before filter: {} after filtering: {}'.format(min_, num_codes, num_codes_after))
    return cpt[t]


def preprocess_notes(text):
    y=re.sub('\\[(.*?)\\]','',text) #remove de-identified brackets y=re.sub('[0-9]+\.','',y)    #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    return y

def filter_diagnoses_codes(diagnoses, min_=5, max_=np.inf):
    t = diagnoses.groupby('ICD9_CODE').ICD9_CODE.transform(len) > min_
    num_codes = len(set(diagnoses['ICD9_CODE']))
    num_codes_after =len(set(diagnoses.loc[t, 'ICD9_CODE']))
    print('removing diagnosis codes occuring less than {} times. \n num codes before filter: {} after filtering: {}'.format(min_, num_codes, num_codes_after))
    return diagnoses[t]

def filter_prescription_codes(med, min_=5, max_=np.inf):
    t = med.groupby('NDC').NDC.transform(len) > min_
    num_codes = len(set(med['NDC']))
    num_codes_after =len(set(med.loc[t, 'NDC']))
    print('removing prescription codes occuring less than {} times. \n num codes before filter: {} after filtering: {}'.format(min_, num_codes, num_codes_after))
    return med[t]


def filter_procedure_codes(procedures, min_=5, max_=np.inf):
    t = procedures.groupby('ICD9_CODE').ICD9_CODE.transform(len) > min_
    num_codes = len(set(procedures['ICD9_CODE']))
    num_codes_after =len(set(procedures.loc[t, 'ICD9_CODE']))
    print('removing procedure codes occuring less than {} times. \n num codes before filter: {} after filtering: {}'.format(min_, num_codes, num_codes_after))
    return procedures[t]

def merge_on_subject(t1, t2):
    return t1.merge(t2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])

def merge_on_subject_admission(t1, t2):
    return t1.merge(t2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def merge_on_subject_admission_left(t1, t2):
    return t1.merge(t2, how='left', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def add_age_to_icustays(stays):
    stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    idxs = stays.AGE < 0
    stays.loc[idxs, 'AGE'] = 90
    return stays
def normalize_column(df, column_name):
    df.loc[:, column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
    return df
def filter_icustays_on_age(stays, min_age=15, max_age=np.inf):
    stays = stays.loc[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays

def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def remove_icu_stays_with_transfer(stays):
    stays = stays.loc[(stays['FIRST_WARDID'] == stays['LAST_WARDID']) & (stays['FIRST_CAREUNIT'] == stays['LAST_CAREUNIT'])]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]

def remove_min_admissions(t, min_admits=1):
    tt = t.groupby('SUBJECT_ID').SUBJECT_ID.transform(len) >= min_admits
    t = t[tt]
    print('num of subjects with min_admits of {} is {}'.format(min_admits, len(set(t['SUBJECT_ID']))))
    return t

def group_by_return_col_list(t, groupby, col, col_name=''):
    if col_name == '':
        col_name = col
    return t.groupby(groupby).apply(lambda x: x[col].values.tolist()).reset_index(name=col_name)
