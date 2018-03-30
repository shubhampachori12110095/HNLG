'''
Do the preprocessing of e2e, opensubs dataset
'''

import sys
import argparse
import pandas as pd
import os
import json
from os.path import join as opj
from tqdm import tqdm

# nltk toolkit
import nltk
import spacy

# Multiprocess
from multiprocessing import Pool as ProcessPool
import multiprocessing

spacy_nlp = spacy.load('en')

def nltk_pos(sent):
    text = nltk.word_tokenize(sent, tagset='universal')
    pos_tag = nltk.pos_tag(text)
    words, tags = [], []
    for w, t in pos_tag:
        words.append(w)
        if t != '.':
            tags.append(t)
        else:
            tags.append('PUNCT')
    return words, tags

def spacy_pos(sent):
    text = spacy_nlp(sent)
    words, tags = [], []
    for word in text:
        words.append(word.text)
        tags.append(word.pos_)
    return words, tags

def both_pos(sent, tolerence=0):
    nltk_word, nltk_tag = nltk_pos(sent)
    spacy_word, spacy_tag = spacy_pos(sent)

    success = False
    if len(set(nltk_tag) - set(spacy_tag)) <= tolerence:
        success = True

    return spacy_word ,spacy_tag, success


class Preprocessor:
    def __init__(self, savedir, tokenizer):
        self.savedir = savedir

        if tokenizer == 'spacy':
            self.tokenizer = spacy_pos
        elif tokenizer == 'nltk':
            self.tokenizer = nltk_pos
        else:
            # default tokenizer = spacy
            self.tokenizer = spacy_pos

    def e2e(self, data_dir):
        # On going
        train = pd.read_csv(opj(data_dir, 'trainset.csv'))
        dev = pd.read_csv(opj(data_dir, 'devset.csv'))

        train_sent = train['ref'].values
        dev_sent = dev['ref'].values

        cpus = multiprocessing.cpu_count()
        workers = ProcessPool(cpus)

        train_dialogues = []
        for words, tags in tqdm(workers.imap(self.tokenizer, train_sent), 
                                    desc='E2E'):
            l1_sent, l2_sent, full = [], [], []
            for word, tag in zip(words, tags):
                full.append(word)
                if tag in ['NOUN', 'PROPN', 'PRON']:
                    l1_sent.append(word)
                    l2_sent.append(word)
                if tag == 'VERB':
                    l2_sent.append(word)
                
    # ----------Opensubs------------
    def process_opensubs_line(self, dialogue):
        q, r = dialogue[0], dialogue[1]
        q_words, q_tags = self.tokenizer(q)
        r_words, r_tags = self.tokenizer(r)

        l1, l2 = [], []
        for word, tag in zip(r_words, r_tags):
            if tag in ['NOUN', 'PROPN', 'PRON']:
                l1.append(word)
                l2.append(word)
            if tag == 'VERB':
                # lemma?
                l2.append(word)
        if len(l1) == 0 or len(l2) == 0:
            return None
        else:
            return {'q':q_words, 'l1':l1, 'l2':l2, 'r':r_words}

    def read_opensubs(self, file):
        # Read raw opensubs
        dialogue, dialogues = [], []
        with open(file, 'r') as f:
            for line in tqdm(f):
                if len(line.strip()) != 0:
                    dialogue.append(line[2:].strip())
                else:
                    dialogues.append(dialogue)
                    dialogue = []

        parsed_dialogues = []
        cpus = multiprocessing.cpu_count()
        workers = ProcessPool(cpus)        
        for parsed_dialogue in tqdm(workers.imap(self.process_opensubs_line,
                                    dialogues), desc='Opensubs:' ):
            if parsed_dialogue != None:
                parsed_dialogues.append(parsed_dialogu)

        return parsed_dialogues



    def opensubs(self, datadir):
        train = opj(datadir, 'opensubs_trial_data_train.txt')
        dev = opj(datadir, 'opensubs_trial_data_dev.txt')
        test = opj(datadir, 'opensubs_trial_data_eval.txt')
        
        train_sentences, dev_sentences, test_sentences = [], [], []
        if os.path.exists(train):
            parsed_train = self.read_opensubs(train)
            json.dump(parsed_train, open(opj(datadir, 'opensubs_train.json'), 'w'), indent=2)
        
        if os.path.exists(dev):
            parsed_dev = self.read_opensubs(dev)
            json.dump(parsed_dev, open(opj(datadir, 'opensubs_dev.json'), 'w'), indent=2)

        if os.path.exists(test):
            parsed_test = self.read_opensubs(test)
            json.dump(parsed_test, open(opj(datadir, 'opensubs_test.json'), 'w'), indent=2)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data preprocessor',
        epilog='Example: python3 preprocess.py --e2e'
    )
    parser.add_argument(
        '--e2e',
        type=str,
        default='/tmp2/exe1023/e2e',
        help='set the datadir of e2e',
    )
    parser.add_argument(
        '--opensubs',
        type=str,
        default='/tmp2/exe1023/opensubs',
        help='set the datadir of opensubs',
    )
    parsed_args = parser.parse_args(sys.argv[1:])
    
    preprocessor = Preprocessor(savedir='./',tokenizer='spacy')
    
    preprocessor.opensubs(parsed_args.opensubs)