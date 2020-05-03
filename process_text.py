import pickle
from nltk.corpus import stopwords
import collections
import string
import os
def preprocess_corpus( corpus):
    sw = set(stopwords.words('english'))
    data = []
    sentences = corpus.split(".")
    s = ""
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence if word not in sw]
        t = " ".join(x)
        s += t + "\n"
    s = s.strip()
    return s

def create_dataset(data, text_type, keys):
    tt = ""
    for j, k in enumerate(keys):
        vv = data[k]
        for v in vv:
            tt += v['text_{}_raw'.format(text_type)]
    return tt

if __name__ == '__main__':

    PATH = './data'
    PATH_DATA = os.path.join(PATH, 'textcode/biobert_pubmed_raw/')
    data = pickle.load(open(os.path.join(PATH_DATA, 'data.pkl'), 'rb'))
    data = data['data']
    train_idx, valid_idx = pickle.load(open(os.path.join(PATH_DATA, 'splits', 'split_0.pkl'), 'rb'))
    text = create_dataset(data, 'discharge', train_idx)
    text = preprocess_corpus(text)
    file = open(os.path.join(PATH, 'input_discharge.txt'), 'w')
    file.writelines(text)
    file.close()
    text = create_dataset(data, 'rest', train_idx)
    text = preprocess_corpus(text)
    file = open(os.path.join(PATH, 'input_rest.txt'), 'w')
    file.writelines(text)
    file.close()
