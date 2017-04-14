import json
from pprint import pprint
import numpy as np
import re
import io
import nltk
nltk.download('punkt')

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, RepeatVector, Activation, Lambda, Flatten, Reshape
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU, AveragePooling1D, Reshape
from keras.layers import GlobalAveragePooling1D
from keras.layers import concatenate, add, dot, Permute
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras import optimizers
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

def loadGloveModel(gloveFile):
    print("Loading Glove Model...")
    f = open(gloveFile,'r', encoding='utf-8')
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index

def tokenize(sent):
    '''Return the tokens of a context including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]

def tokenizeVal(sent):
    '''Return the tokens of a context including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    tokenizedSent = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]
    tokenIdx2CharIdx = [None] * len(tokenizedSent)
    idx = 0
    token_idx = 0
    while idx < len(sent) and token_idx < len(tokenizedSent):
        word = tokenizedSent[token_idx]
        if sent[idx:idx+len(word)] == word:
            tokenIdx2CharIdx[token_idx] = idx
            idx += len(word)
            token_idx += 1
        else:
            idx += 1
    return tokenizedSent, tokenIdx2CharIdx


def splitDatasets(f):
    '''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices,
       and keep track of max context and question lengths.
    '''
    xContext = [] # list of contexts paragraphs
    xQuestion = [] # list of questions
    xQuestion_id = [] # list of question id
    xAnswerBegin = [] # list of indices of the beginning word in each answer span
    xAnswerEnd = [] # list of indices of the ending word in each answer span
    xAnswerText = [] # list of the answer text
    maxLenContext = 0
    maxLenQuestion = 0

    for data in f['data']:
        paragraphs = data['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            contextTokenized = tokenize(context.lower())
            contextLength = len(contextTokenized)
            if contextLength > maxLenContext:
                maxLenContext = contextLength
            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = tokenize(question.lower())
                if len(questionTokenized) > maxLenQuestion:
                    maxLenQuestion = len(questionTokenized)
                question_id = qa['id']
                answers = qa['answers']
                for answer in answers:
                    answerText = answer['text']
                    answerTokenized = tokenize(answerText.lower())
                    # find indices of beginning/ending words of answer span among tokenized context
                    contextToAnswerFirstWord = context1[:answer['answer_start'] + len(answerTokenized[0])]
                    answerBeginIndex = len(tokenize(contextToAnswerFirstWord.lower())) - 1
                    answerEndIndex = answerBeginIndex + len(answerTokenized) - 1

                    xContext.append(contextTokenized)
                    xQuestion.append(questionTokenized)
                    xQuestion_id.append(str(question_id))
                    xAnswerBegin.append(answerBeginIndex)
                    xAnswerEnd.append(answerEndIndex)
                    xAnswerText.append(answerText)
    return xContext, xQuestion, xQuestion_id, xAnswerBegin, xAnswerEnd, xAnswerText, maxLenContext, maxLenQuestion


def vectorizeData(xContext, xQuestion, xAnswerBeing, xAnswerEnd, word_index, context_maxlen, question_maxlen):
    '''Vectorize the words to their respective index and pad context to max context length and question to max question length.
       Answers vectors are padded to the max context length as well.
    '''
    X = []
    Xq = []
    YBegin = []
    YEnd = []
    for i in range(len(xContext)):
        x = [word_index[w] for w in xContext[i]]
        xq = [word_index[w] for w in xQuestion[i]]
        # map the first and last words of answer span to one-hot representations
        y_Begin =  np.zeros(len(xContext[i]))
        y_Begin[xAnswerBeing[i]] = 1
        y_End = np.zeros(len(xContext[i]))
        y_End[xAnswerEnd[i]] = 1
        X.append(x)
        Xq.append(xq)
        YBegin.append(y_Begin)
        YEnd.append(y_End)
    return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post'), pad_sequences(YBegin, maxlen=context_maxlen, padding='post'), pad_sequences(YEnd, maxlen=context_maxlen, padding='post')

def import_json(json_file):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    return data

# Note: Need to download and unzip Glove pre-train model files into same file as this script
GloveDimOption = '50' # this  could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
embeddings_index = loadGloveModel('./data/glove/glove.6B.' + GloveDimOption + 'd.txt')

# load training data, parse, and split
print('Loading in training data...')
trainData = import_json('./data/squad/train_small.json')
tContext, tQuestion, tQuestion_id, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = splitDatasets(trainData)

# load validation data, parse, and split
print('Loading in validation data...')
valData = import_json('./data/squad/val_small.json')
vContext, vQuestion, vQuestion_id, vAnswerBegin, vAnswerEnd, vAnswerText, maxLenVContext, maxLenVQuestion = splitDatasets(valData)

print('Building vocabulary...')
# build a vocabular over all training and validation context paragraphs and question words
vocab = {}
for words in tContext + tQuestion + vContext + vQuestion:
    for word in words:
        if word not in vocab:
            vocab[word] = 1
vocab = sorted(vocab.keys())
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_index = dict((c, i + 1) for i, c in enumerate(vocab))
context_maxlen = max(maxLenTContext, maxLenVContext)
question_maxlen = max(maxLenTQuestion, maxLenVQuestion)

# vectorize training and validation datasets
print('Begin vectoring process...')

#tX: training Context, tXq: training Question, tYBegin: training Answer Begin ptr, tYEnd: training Answer End ptr
tX, tXq, tYBegin, tYEnd = vectorizeData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index, context_maxlen, question_maxlen)

# shuffle train data
randindex = np.random.permutation(tX.shape[0])
tX = tX[randindex, :]
tXq = tXq[randindex, :]
tYBegin = tYBegin[randindex, :]
tYEnd = tYEnd[randindex, :]

# vX: validation Context, vXq: validation Question
vX, vXq, vYBegin, vYEnd = vectorizeData(vContext, vQuestion, vAnswerBegin, vAnswerEnd, word_index, context_maxlen, question_maxlen)
print('Vectoring process completed.')

print('tX.shape = {}'.format(tX.shape))
print('tXq.shape = {}'.format(tXq.shape))
print('tYBegin.shape = {}'.format(tYBegin.shape))
print('tYEnd.shape = {}'.format(tYEnd.shape))
print('vX.shape = {}'.format(vX.shape))
print('vXq.shape = {}'.format(vXq.shape))
print('context_maxlen, question_maxlen = {}, {}'.format(context_maxlen, question_maxlen))

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = len(word_index)
EMBEDDING_DIM = int(GloveDimOption)
MAX_SEQUENCE_LENGTH = context_maxlen

embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Hyper-parameters
CONTEXT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 256
EPOCHS = 100

context = Input(shape=(context_maxlen,), dtype='int32')
encoded_context = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                            weights=[embedding_matrix],
                            input_length=context_maxlen, trainable=False)(context)
encoded_context = Dropout(0.3)(encoded_context)

question = Input(shape=(question_maxlen,), dtype='int32')
encoded_question = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                             weights=[embedding_matrix],
                             input_length=question_maxlen, trainable=False)(question)
encoded_question = Dropout(0.3)(encoded_question)
encoded_question = LSTM(EMBEDDING_DIM)(encoded_question)
encoded_question = RepeatVector(context_maxlen)(encoded_question)

merged = add([encoded_context, encoded_question])
merged = LSTM(EMBEDDING_DIM)(merged)
merged = Dropout(0.3)(merged)

answerPtrBegin_output = Dense(context_maxlen, activation='softmax')(merged)
Lmerge = concatenate([merged, answerPtrBegin_output], name='merge2')
answerPtrEnd_output = Dense(context_maxlen, activation='softmax')(Lmerge)

model = Model([context, question], [answerPtrBegin_output, answerPtrEnd_output])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print('Training...')
model.fit([tX, tXq], [tYBegin, tYEnd],
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2,
          verbose=1)
predictions = model.predict([vX, vXq], batch_size=64)

print(predictions[0].shape, predictions[1].shape)
# make class prediction
ansBegin = np.zeros((predictions[0].shape[0],), dtype=np.int32)
ansEnd = np.zeros((predictions[0].shape[0],),dtype=np.int32)
for i in range(predictions[0].shape[0]):
	ansBegin[i] = predictions[0][i, :].argmax()
	ansEnd[i] = predictions[1][i, :].argmax()
print(ansBegin.min(), ansBegin.max(), ansEnd.min(), ansEnd.max())

for i in range(predictions[0].shape[0]):
    print(' '.join(vQuestion[i]))
    print('Predicted Answer:', ' '.join(vContext[i][ansBegin[i] : ansEnd[i] + 1]))
    # print(vAnswerBegin[i], vAnswerEnd[i])
    print('True Answer:', ' '.join(vContext[i][vAnswerBegin[i] : vAnswerEnd[i] + 1]))
    print()