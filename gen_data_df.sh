#!/bin/bash
python gen_data_df.py -p ./data/mimic3 -s ./data/ -min-adm 1 -ft "Discharge summary" Physician ECG Radiology Respiratory Nursing Pharmacy Nutrition
