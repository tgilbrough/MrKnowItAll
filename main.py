import argparse
import numpy as np
import tensorflow as tf
import json
import os

import data
from model import Model 

def get_parser():
    parser = argparse.ArgumentParser()
    home = "" #os.path.expanduser("~")
    parser.add_argument('--keep_prob', default=0.8)
    parser.add_argument('--hidden_size', default=100)
    parser.add_argument('--dim_size', default=50) # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--train_path', default='./data/msmarco/train/location.json')
    parser.add_argument('--val_path', default='./data/msmarco/dev/location.json')
    parser.add_argument('--reference_path', default='./eval/references.json')
    parser.add_argument('--candidate_path', default='./eval/candidates.json')
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--batch_size', default=128)

    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()

    print('Preparing embedding matrix.')
    
    # load training data, parse, and split
    print('Loading in training data...')
    trainData = data.importMsmarco(config.train_path)
    tContext, tQuestion, tQuestionID, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = data.splitMsmarcoDatasets(trainData)

    # load validation data, parse, and split
    print('Loading in validation data...')
    valData = data.importMsmarco(config.val_path)
    vContext, vQuestion, vQuestionID, vAnswerBegin, vAnswerEnd, vAnswerText, maxLenVContext, maxLenVQuestion = data.splitMsmarcoDatasets(valData)

    print('Building vocabulary...')
    # build a vocabular over all training and validation context paragraphs and question words
    vocab = data.buildVocab(tContext + tQuestion + vContext + vQuestion)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_index = dict((c, i + 1) for i, c in enumerate(vocab))
    config.max_context_size = max(maxLenTContext, maxLenVContext)
    config.max_ques_size = max(maxLenTQuestion, maxLenVQuestion)

    # Note: Need to download and unzip Glove pre-train model files into same file as this script
    embeddings_index = data.loadGloveModel('./data/glove/glove.6B.' + str(config.dim_size) + 'd.txt')
    config.emb_mat = data.createEmbeddingMatrix(embeddings_index, word_index)

    # vectorize training and validation datasets
    print('Begin vectoring process...')

    #tX: training Context, tXq: training Question, tYBegin: training Answer Begin ptr, tYEnd: training Answer End ptr
    tX, tXq, tYBegin, tYEnd = data.vectorizeData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index, config.max_context_size, config.max_ques_size)
    vX, vXq, vYBegin, vYEnd = data.vectorizeData(vContext, vQuestion, vAnswerBegin, vAnswerEnd, word_index, config.max_context_size, config.max_ques_size)

    print('Vectoring process completed.')

    model = Model(config)

if __name__ == "__main__":
    main()