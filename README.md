# TAPER-EHR code (Electronic Health Record Representation Learning)
This repository contains the code for [(arxiv link)](https://arxiv.org/abs/1908.03971)

```
article{darabi2019taper,
  title={TAPER: Time-Aware Patient EHR Representation},
  author={Darabi, Sajad and Kachuee, Mohammad and Fazeli, Shayan and Sarrafzadeh, Majid},
  journal={arXiv preprint arXiv:1908.03971},
  year={2019}
}
```

## Dependencies

This project in a conda virtual environment on Ubuntu 16.04 with CUDA 10.0. Dependencies:
* [pytorch with torchvision](http://pytorch.org/). Note that you need at least a fairly recent version of PyTorch (e.g., 1.0). 
* [tensorflow with tensorboard](https://www.tensorflow.org/install/) (This is not critical, however, and you can comment out all references to TF and TB) 

Checkout the `requirements.txt` file.


## The Pretrained Bert Model

Bert models are typically trained in tensorflow and hence require to be ported into pytorch. 

For example, download the [Biobert-Base v1.1](https://github.com/naver/biobert-pretrained)

Once you've downloaded the model to convert into pytorch run the following from ./model/bert_things/pytorch_pretrained_bert/ folder

```
python convert_to_pytorch.py --path <path-to-biobert-folder>/biobert_model.ckpt --config <path-to-biobert-folder>/bert_config.json --save <path-to-save-converted-model>
```

Once the pretrained model has been converted we can load it into our bert models in pytorch. 

## Running the experiments

Make sure you have downloaded the MIMIC III dataset (specifically the csv files). Once you have downloaded the dataset
run the following to compile a dataframe where rows correspond to admissions:

```
python gen_data_df.py -p <path-to-mimic-folder> -s <path-to-save-files> -min-adm 1 -ft "Discharge summary" Physician ECG Radiology Respiratory Nursing Pharmacy Nutrition
```

The above will generate a dataframe containing demographics, medical codes, and medical texts for each admission. The `min-adm` argument is used to filter out patients with less than the specified number of admissions and `ft` argument is used to filter texts that are included in the patients row . Next, we will generate a dictionary containing patient id's and values list of dicts containing admission data/labels.

```
python gen_text_plus_code_data.py -p '<path-to-where-df-were-saved>/df_less_1.pkl' -s '<path-to-save-output>' -et -cpb './data/biobert_pubmed/bert_config.json' -sdp './data/biobert_pubmed_model' -vpb './data/biobert_pubmed/vocab.txt' -bsl 512 -diag -proc -med -cpt -sc -vp <path-to-medical-code-vocabulary>
```

In the above command 
    - et is a flag to embed text
    - cbp is the path to pretrained bert config
    - sdp is the path to pretrained state_dict
    - vbp is the path to vocab on which bert was trained on
    - bsl is the maximum sequence length
    - diag is a flag to add diagnoses to final output
    - proc is a flag to add procedure to final output
    - med is a flag to add medical codes to final output
    - cpt is a flag to add cpt to final output
    - sc is a flag for short codes if you specify this you must add -vp
    - vp path where diagnoses and cpt map code to id are stored (take a look at [here](https://github.com/sajaddarabi/HCUP-US-EHR) )

Once the above script finishes, we can pretrain both code and text models:


### Running Code Training

To run the code pretraining step there are example config files in ./configs/codes

For example to run a pretraining step for diagnoses codes run the following from train.py directory

`python train.py -c ./configs/codes/diag.json`


### Running Text Training

To run the code summarizer training step there are example config files in ./configs/text

For example to train the text summarizer on discharge run the following (make sure the paths are correctly set in the config file)

`python train.py -c ./configs/text/pubmed_discharge.json`


### Running Classification

To run classification runs examples config files are in ./configs/taper.
For example to run the mortality task run

`python train.py -c ./configs/taper/red.json`

## Acknowledgments

This repository includes code from:
* The [huggingface transformer very initial commits](https://github.com/huggingface/transformers).
* TensorboardLogger from [this gist](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514) 
