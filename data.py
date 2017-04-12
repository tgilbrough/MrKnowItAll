import os
import numpy as np
import json
from nltk.tokenize import word_tokenize

def load_data(categories, train_dir):
    queries = []

    for category in categories:
        file_count = 0
        file_path = os.path.join(train_dir, category + '_' + str(file_count) + '.json')
        while os.path.isfile(file_path):
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    queries.append(data)            

            file_count += 1
            file_path = os.path.join(train_dir, category + '_' + str(file_count) + '.json')

    return queries

def load_glove_model(glove_file):
    print("Loading Glove Model")
    model = {}
    with open(glove_file,'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = [float(val) for val in split_line[1:]]
            model[word] = embedding

    print("Done, ",len(model)," words loaded")
    return model

def get_wordvec(word, model, n):
    if word in model:
        return model[word]
    else:
        # TODO: Find better solution
        # For unknown words
        return np.zeros(n)
