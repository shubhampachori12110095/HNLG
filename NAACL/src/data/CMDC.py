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


def CMDC(data_dir, is_spacy, is_lemma, use_punct, min_length=-1):
    # Step 1: Get the raw dialogues from data files
    raw_dialogues = parse_data(data_dir)
    # Step 2: Parse the dialogues
    dialogues = parse_dialogues(raw_dialogues, is_spacy)
    # Step 3: Build the input data and output labels
    input_data, output_labels = \
        build_dataset(dialogues, is_lemma, use_punct, min_length)
    return input_data, output_labels


def parse_data(data_dir):
    lines_file = os.path.join(data_dir, "movie_lines.txt")
    conversations_file = os.path.join(data_dir, "movie_conversations.txt")
    raw_lines = {}
    with open(lines_file, 'r', encoding='ISO-8859-2') as file:
        for line in file:
            data = line.split("+++$+++")
            line_id, line_text = data[0], data[4]
            raw_lines[line_id.strip()] = line_text.strip()

    raw_dialogues = []
    with open(conversations_file, 'r', encoding='ISO-8859-2') as file:
        for line in file:
            line_indices = ast.literal_eval(line.split("+++$+++")[3].strip())
            raw_dialogues.append([raw_lines[idx] for idx in line_indices])
    del(raw_lines)
    return raw_dialogues


def parse_dialogues(raw_dialogues, is_spacy):
    dialogues = []
    if is_spacy:
        spacy_parser = spacy.load('en')
    else:
        nltk_lemmatizer = WordNetLemmatizer()
    for idx, dialog in enumerate(raw_dialogues):
        if idx % 1000 == 0:
            print_time_info(
                "Processed {}/{} dialogues".format(idx, len(raw_dialogues)))
        spacy_parsed_dialog = []
        nltk_parsed_dialog = []
        for line in dialog:
            spacy_line, nltk_line = [], []
            if is_spacy:
                parsed_line = spacy_parser(line)
                spacy_line = [
                    d for d in [
                        (word.text, word.pos_)
                        for word in parsed_line
                        ] if d[0] != ' '
                    ]
                spacy_parsed_dialog.append(spacy_line)
            else:
                nltk_line = pos_tag(word_tokenize(line), tagset='universal')
                nltk_line = [
                        (d[0], d[1])
                        if d[1] != '.'
                        else (d[0], 'PUNCT') for d in nltk_line]
                nltk_parsed_dialog.append(nltk_line)

        if spacy_parsed_dialog != []:
            dialogues.append(spacy_parsed_dialog)
        else:
            dialogues.append(nltk_parsed_dialog)
    del(raw_dialogues)
    return dialogues


def build_dataset(dialogues, is_lemma, use_punct, min_length):
    input_data = []
    output_labels = [[] for _ in range(4)]
    spacy_parser = spacy.load('en')
    """
        For now, the data has four different layers:
            1. NOUN + PROPN + PRON
            2. NOUN + PROPN + PRON + VERB
            3. NOUN + PROPN + PRON + VERB + ADJ + ADV
            4. ALL
    """
    for idx, dialog in enumerate(dialogues):
        if idx % 1000 == 0:
            print_time_info(
                    "Parsed {}/{} dialogues".format(idx, len(dialogues)))
        for idx in range(len(dialog)-1):
            input_data.append([
                word[0].lower()
                for word in dialog[idx]
                if (word[1] != 'PUNCT' or use_punct == 1)])
            output_label = [[] for _ in range(4)]
            for w in dialog[idx+1]:
                if w[1] in ['NOUN', 'PROPN', 'PRON']:
                    output_label[0].append(w[0].lower())
                    output_label[1].append(w[0].lower())
                    output_label[2].append(w[0].lower())
                    output_label[3].append(w[0].lower())
                elif w[1] == 'VERB':
                    word = w[0].lower()
                    if is_lemma:
                        word = spacy_parser(word)[0].lemma_
                    output_label[1].append(word)
                    output_label[2].append(word)
                    output_label[3].append(word)
                elif w[1] in ['ADJ', 'ADV']:
                    output_label[2].append(w[0].lower())
                    output_label[3].append(w[0].lower())
                else:
                    if w[1] == "PUNCT" and not use_punct:
                        continue
                    output_label[3].append(w[0].lower())

            for idx in range(4):
                output_labels[idx].append(output_label[idx])

    if min_length == -1:
        print_time_info(
                "No minimal length, data count: {}".format(len(dialogues)))
    else:
        print_time_info("Minimal length is {}".format(min_length))
        idxs = []
        for idx, sent in enumerate(input_data):
            if len(output_labels[3][idx]) > min_length:
                idxs.append(idx)
        input_data = [input_data[i] for i in idxs]
        output_labels = [
                [output_label[i] for i in idxs]
                for output_label in output_labels]
        print_time_info("Data count: {}".format(len(idxs)))
    return input_data, output_labels
