#!/bin/bash
python convert_to_pytorch.py --path ../../data/pretrained_bert_tf/biobert_pretrain_output_disch_100000/model.ckpt-100000 --config ../../data/pretrained_bert_tf/biobert_pretrain_output_disch_100000/bert_config.json --save ../../data/bert_disch_notes
