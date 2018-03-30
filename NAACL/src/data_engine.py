"""
Usage:
    You should put the movie dialog dataset under a specific directory,
    like data/CMDC, then initialize the data engine like below:

    data_engine = DataEngine("data/CMDC", "CMDC", save_path="data.pkl")

    You can also change the value of is_spacy (use spacy or use nltk instead)
    and is_lemma(whether lemmatize the verbs or not)
"""

from torch.utils.data import Dataset
from nltk.corpus import wordnet, treebank
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
import os
from os.path import join as opj
import ast
import random
import pickle
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from utils import print_time_info
from tokenizer import Tokenizer
from text_token import _UNK, _PAD, _BOS, _EOS
from data.CMDC import CMDC
from data.E2ENLG import E2ENLG
from data.REPEATSEQ import REPEATSEQ


class DataEngine(Dataset):
    def __init__(
            self,
            data_dir,
            dataset,
            save_path='data.pkl',
            vocab_path='vocab.pkl',
            is_spacy=True,
            is_lemma=True,
            fold_attr=True,
            use_punct=False,
            vocab_size=20000,
            n_layers=4,
            min_length=5,
            en_max_length=None,
            de_max_length=None,
            regen=False,
            train=True
            # partition_ratio=0.95
    ):
        if is_spacy:
            self.spacy_parser = spacy.load('en')
            print_time_info("Use Spacy as the parser")
        else:
            self.nltk_lemmatizer = WordNetLemmatizer()
            print_time_info("Use NLTK as the parser")
        self.is_spacy = is_spacy
        self.is_lemma = is_lemma
        self.fold_attr = fold_attr
        self.use_punct = use_punct
        self.data_dir = data_dir
        self.save_path = save_path
        self.vocab_path = vocab_path
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dataset = dataset
        self.min_length = min_length
        self.en_max_length = en_max_length if en_max_length else -1
        self.de_max_length = de_max_length if de_max_length else -1
        self.regen = regen
        if self.dataset in ["CMDC", "OPENSUBS", "REPEATSEQ"]:
            self.split_vocab = False
        else:
            self.split_vocab = True
        self.tokenizer = Tokenizer(vocab_path, self.split_vocab, regen, train)
        self.counter = 0
        self.train = train
        # self.partition_ratio = partition_ratio
        self.prepare_data()
        # TODO: Add support for more decoder layers

    def prepare_data(self):
        if not os.path.exists(self.save_path) or self.regen:
            if self.regen:
                print_time_info("Regenerate the data...")
            else:
                print_time_info("There isn't any usable save...")
            if not os.path.isdir(self.data_dir):
                print_time_info("Error: The dataset doesn't exist")
                exit()
            print_time_info("Start reading dataset {} from {}".format(
                self.dataset, self.data_dir))
            if self.dataset == "CMDC":
                self.input_data, self.output_labels = CMDC(
                        self.data_dir, self.is_spacy, self.is_lemma,
                        self.use_punct, self.min_length)
            elif self.dataset == "E2ENLG":
                self.input_data, self.output_labels = E2ENLG(
                        self.data_dir, self.is_spacy, self.is_lemma,
                        self.fold_attr, self.use_punct,
                        self.min_length, self.train)
            elif self.dataset == "REPEATSEQ":
                self.input_data, self.output_labels = REPEATSEQ(self.data_dir)
            elif self.dataset == "DSTC6":
                self.DSTC6()
            elif self.dataset == "DBDC3":
                self.DBDC3()
            elif self.dataset == 'OPENSUBS':
                self.OPENSUBS()
        else:
            self.input_data, self.output_labels = \
                    pickle.load(open(self.save_path, 'rb'))
            print_time_info("Load the data from {}".format(self.save_path))

        if not os.path.exists(self.vocab_path) or (self.regen and self.train):
            self.build_vocab()
        if not os.path.exists(self.save_path) or self.regen:
            self.tokenize_sents()
            self.crop()
            pickle.dump(
                    [self.input_data, self.output_labels],
                    open(self.save_path, 'wb'))
            print_time_info(
                    "Create the save file {}".format(self.save_path))

        # shrink the vocab to vocab size
        self.tokenizer.shrink_vocab(self.vocab_size)
        self.add_unk()

        # pick the labels for different n_layers
        if self.n_layers == 1:
            self.output_labels = [self.output_labels[3]]
        elif self.n_layers == 2:
            self.output_labels = [self.output_labels[1], self.output_labels[3]]

        # partition training and testing data
        """
        train_data_len = int(len(self.input_data) * self.partition_ratio)
        self.input_data = np.array(self.input_data)
        self.output_labels = np.array(self.output_labels)

        if self.train:
            self.input_data = self.input_data[:train_data_len]
            self.output_labels = self.output_labels[:, :train_data_len]
        else:
            self.input_data = self.input_data[train_data_len:]
            self.output_labels = self.output_labels[:, train_data_len:]
        """
    # NEED FIX (not urgent though)
    def OPENSUBS(self):
        dirpath = os.getenv('OPENSUBS')
        if dirpath is None:
            # only on cl5)
            dirpath = '/tmp2/exe1023/opensubs'
        train = json.load(open(opj(dirpath, 'opensubs_train.json'), 'r'))
        self.data = []
        for sample in tqdm(train):
            self.data.append((sample['q'], [sample['l1'], sample['l2']]))

    def build_vocab(self):
        if not self.split_vocab:
            corpus = []
            for sent in self.input_data:
                corpus.extend(sent)
            for sent in self.output_labels[-1]:
                corpus.extend(sent)
            self.tokenizer.build_vocab(corpus)
        else:
            corpus = []
            for sent in self.output_labels[-1]:
                corpus.extend(sent)
            tokens = []
            for attrs in self.input_data:
                tokens.extend(attrs)
            self.tokenizer.build_vocab(corpus, tokens)

    def tokenize_sents(self):
        for idx, sent in enumerate(self.input_data):
            self.input_data[idx] = self.tokenizer.tokenize(sent, True)
        for idx, output_labels in enumerate(self.output_labels):
            for sidx, sent in enumerate(output_labels):
                self.output_labels[idx][sidx] = \
                        self.tokenizer.tokenize(sent, False)
                self.output_labels[idx][sidx].append(_EOS)

    def add_unk(self):
        if not self.split_vocab:
            for idx, sent in enumerate(self.input_data):
                for w_idx, word in enumerate(sent):
                    # 4 for special token
                    if word >= self.vocab_size + 4:
                        self.input_data[idx][w_idx] = _UNK

        for l_idx, labels in enumerate(self.output_labels):
            for idx, sent in enumerate(labels):
                for w_idx, word in enumerate(sent):
                    # 4 for special token
                    if word >= self.vocab_size + 4:
                        self.output_labels[l_idx][idx][w_idx] = _UNK

    def crop(self):
        if self.en_max_length != -1:
            for idx, sent in enumerate(self.input_data):
                self.input_data[idx] = sent[:self.en_max_length]
        if self.de_max_length != -1:
            for idx, labels in enumerate(self.output_labels):
                for sidx, sent in enumerate(labels):
                    self.output_labels[idx][sidx] = sent[:self.de_max_length]

    def expand(self):
        inter_labels = [[], [], [], []]
        for l_idx, labels in enumerate(self.output_labels[:-1]):
            for s_idx, sent in enumerate(labels):
                upper_sent = self.output_labels[l_idx+1][s_idx]
                inter_label = []
                lidx, hidx = 0, 0
                while lidx != len(sent):
                    inter_label.append(sent[lidx])
                    if sent[lidx] == upper_sent[hidx]:
                        lidx += 1
                    hidx += 1
                inter_labels[l_idx+1].append(inter_label)

        for _ in range(len(self.input_data)):
            inter_labels[0].append([_UNK])

        return inter_labels

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)

    def untokenize(self, sent, is_token=False):
        return self.tokenizer.untokenize(sent, is_token)

    def DSTC6(self):
        pass

    def DBDC3(self):
        pass

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (
            self.input_data[idx],
            [labels[idx] for labels in self.output_labels])


if __name__ == '__main__':
    # test
    engine = DataEngine('cmdc', 'OPENSUBS')
