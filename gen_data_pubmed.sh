#!/bin/bash
python gen_text_plus_code_data.py -p './data/textcode/df_less_1.pkl' -s './data/textcode/biobert_pubmed_raw' -et -cpb './data/biobert_pubmed/bert_config.json' -sdp './data/biobert_pubmed_model' -vpb './data/biobert_pubmed/vocab.txt' -bsl 512 -diag -proc -med -cpt -sc -vp './data/' #-tsld 27 -tslr 33
