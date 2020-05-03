from collections import Counter, OrderedDict
import pickle
__all__ = ['Vocab']

class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, vocab_file=None, *kargs, **kwargs):
        self.counter = Counter()
        self.special = special
        self.vocab_file = vocab_file
        self.min_freq = min_freq
        self.max_size = max_size
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        self.unknown_code = -1

    def convert_to_ids(self, codes, code_type='D', short_icd9=False):
        '''mimic-iii format
        '''
        if (short_icd9):
            convert = lambda x: self.convert_to_3digit_icd9(x, code_type)
            icd9 = list(map(convert, codes))
        else:
            convert = lambda x: self.convert_to_icd9(x, code_type)
            icd9 = list(map(convert, codes))

        self.counter.update(icd9)
        self.add_codes(icd9)

        return self.get_indices(icd9)

    def build_vocab(self):
        if self.vocab_file:
            self._build_from_file(self.vocab_file)
        else:
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for code, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_feq: break
                self.add_code(code)

    def _build_from_file(self, vocab_path, text_file=False):
        if text_file:
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                for l in f:
                    codes = l.strip().split()
                    self.add_codes(codes)
        else:
            if (hasattr(self, 'sym2idx')):
                self.sym2idx = {**self.sym2idx, **pickle.load(open(vocab_path, 'rb'))}
            else:
                self.sym2idx = pickle.load(open(vocab_path, 'rb'))

            if (hasattr(self, 'idx2sym')):
                self.idx2sym += list(set([v for k, v in self.sym2idx.items()]))
            else:
                self.idx2sym = list(set([v for k, v in self.sym2idx.items()]))


    def _load_tok_names(self, tok_names_path):
        self.tok_names = pickle.load(open(tok_names_path, 'rb'))

    def convert_to_icd9(self, dxStr, ext):
        if (dxStr in self.sym2idx.keys()):
            return dxStr

        if (type(dxStr) != str):
            dxStr = str(dxStr)
        if dxStr.startswith('E'):
            if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
            else: return ext + '_' + dxStr
        else:
            if len(dxStr) > 3: return ext + '_' + dxStr[:3] + '.' + dxStr[3:]
            else: return ext + '_' + dxStr

    def convert_to_3digit_icd9(self, dxStr, ext):
        if (dxStr in self.sym2idx.keys()):
            return dxStr

        if (type(dxStr) != str):
            dxStr = str(dxStr)
        if dxStr.startswith('E'):
            if len(dxStr) > 4: return ext + '_' + dxStr[:4]
            else: return ext + '_' + dxStr
        else:
            if len(dxStr) > 3: return ext + '_' + dxStr[:3]
            else: return ext + '_' + dxStr

    def add_code(self, code):
        if code is None:
            return
        if code not in self.sym2idx:
            self.idx2sym.append(code)
            self.sym2idx[code] = len(self) - 1

    def add_codes(self, codes):
        for c in codes:
            if c is not None:
                self.add_code(c)

    def get_idx(self, code):
        if code in self.sym2idx:
            return self.sym2idx[code]
        else:
            return self.unknown_code

    def get_indices(self, codes):
        return [self.get_idx(c) for c in codes]

    def __len__(self):
        return len(self.idx2sym)
