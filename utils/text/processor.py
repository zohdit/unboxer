

import random
from config.config_data import EXPECTED_LABEL
from feature_map.imdb.predictor import Predictor
import numpy as np

from config.config_featuremaps import INPUT_MAXLEN
from utils import global_values


def process_text_contributions(data, contributions):
    """
    process contributions by transforming the text to vec 
    and assing contributions to each word
    :param data: original texts
    :param contributions: text contributions
    :return: The processed contributions
    """
    processed_contribution = np.zeros(shape=(len(data), INPUT_MAXLEN), dtype=float)
    for idx1 in range(len(data)):
        for idx2 in range(len(data[idx1])):
            word_index = data[idx1][idx2]
            processed_contribution[idx1][word_index] = contributions[idx1][idx2] 
    return processed_contribution    

def words_from_contribution(text, contribution):
    """
    extract important word from the contribution
    :param contributions: text contribution
    :return: list of important words
    """

    seq = Predictor.tokenizer.texts_to_sequences([text]) 
    generated_text = Predictor.tokenizer.sequences_to_texts(seq)

    sequence = seq[0]
    text = generated_text[0].split()
    list_words = []


    # iterate on processed contribution which is a vector with INPUT_MAXLEN size
    for idx1 in range(len(contribution)):
        # check if the word has positive contribution
        if contribution[idx1] > 0:
            # find the corresponding word and add it to the list of important words if its not already added
            word = [text[i] for i, item in enumerate(sequence) if item == idx1 and item != []]
            if word != []:
                word = word[0]
                if [word] not in list_words:
                    list_words.append(word)

    return list_words  


def top_ten(text, contribution):
    """
    extract important word from the contribution
    :param contributions: text contribution
    :return: list of important words
    """
    
    seq = Predictor.tokenizer.texts_to_sequences([text]) 
    generated_text = Predictor.tokenizer.sequences_to_texts(seq)

    sequence = seq[0]
    text = generated_text[0].split()
    list_words = []


    # iterate on processed contribution which is a vector with INPUT_MAXLEN size
    for idx1 in range(len(contribution)):
        # check if the word has positive contribution
        if contribution[idx1] > 0:
            # find the corresponding word and add it to the list of important words if its not already added
            word = [text[i] for i, item in enumerate(sequence) if item == idx1 and item != []]
            if word != []:
                word = word[0]
                if [word] not in list_words:
                    list_words.append([word, contribution[idx1]])
    list_words = sorted(list_words, key=lambda tup: tup[1], reverse=True)[:10] 
    return list_words  

def random_words():
    words = []
    num = random.randint(10, 30)
    words = random.sample(list(Predictor.tokenizer.word_index.keys()), num)    
    return words
