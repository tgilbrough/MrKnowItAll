import numpy as np
import tensorflow as tf

from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, RepeatVector, Activation
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU
from keras.layers import concatenate, add
from keras.models import Model

import data

EMBEDDING_DIM = 50 # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
TRAINING_DATA_PATH = './data/msmarco/train/location.json' #'./data/squad/train_small.json'
VAL_DATA_PATH = './data/msmarco/dev/location.json'

# load training data, parse, and split
print('Loading in training data...')
trainData = data.importMsmarco(TRAINING_DATA_PATH)
tContext, tQuestion, tQuestionID, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = data.splitMsmarcoDatasets(trainData)

# load validation data, parse, and split
print('Loading in validation data...')
valData = data.importMsmarco(VAL_DATA_PATH)
vContext, vQuestion, vQuestionID, vAnswerBegin, vAnswerEnd, vAnswerText, maxLenVContext, maxLenVQuestion = data.splitMsmarcoDatasets(valData)

print('Building vocabulary...')
# build a vocabular over all training and validation context paragraphs and question words
vocab = data.buildVocab(tContext + tQuestion + vContext + vQuestion)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_index = dict((c, i + 1) for i, c in enumerate(vocab))
context_maxlen = max(maxLenTContext, maxLenVContext)
question_maxlen = max(maxLenTQuestion, maxLenVQuestion)

# vectorize training and validation datasets
print('Begin vectoring process...')

#tX: training Context, tXq: training Question, tYBegin: training Answer Begin ptr, tYEnd: training Answer End ptr
tX, tXq, tYBegin, tYEnd = data.vectorizeData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index, context_maxlen, question_maxlen)
vX, vXq, vYBegin, vYEnd = data.vectorizeData(vContext, vQuestion, vAnswerBegin, vAnswerEnd, word_index, context_maxlen, question_maxlen)

print('Vectoring process completed.')

# shuffle train data
randindex = np.random.permutation(tX.shape[0])
tX = tX[randindex, :]
tXq = tXq[randindex, :]
tYBegin = tYBegin[randindex, :]
tYEnd = tYEnd[randindex, :]

print('tX.shape = {}'.format(tX.shape))
print('tXq.shape = {}'.format(tXq.shape))
print('tYBegin.shape = {}'.format(tYBegin.shape))
print('tYEnd.shape = {}'.format(tYEnd.shape))
print('vX.shape = {}'.format(vX.shape))
print('vXq.shape = {}'.format(vXq.shape))
print('context_maxlen, question_maxlen = {}, {}'.format(context_maxlen, question_maxlen))

print('Preparing embedding matrix.')

# Note: Need to download and unzip Glove pre-train model files into same file as this script
embeddings_index = data.loadGloveModel('./data/glove/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt')

# prepare embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
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

question = Input(shape=(question_maxlen,), dtype='int32')
encoded_question = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                             weights=[embedding_matrix],
                             input_length=question_maxlen, trainable=False)(question)
encoded_question = LSTM(EMBEDDING_DIM)(encoded_question)
encoded_question = RepeatVector(context_maxlen)(encoded_question)

merged = add([encoded_context, encoded_question])
merged = LSTM(64)(merged)

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

for i in range(predictions[0].shape[0]):
    print(' '.join(vQuestion[i]))
    print('Predicted Answer:', ' '.join(vContext[i][ansBegin[i] : ansEnd[i]]))
    print('True Answer:', ' '.join(vContext[i][vAnswerBegin[i] : vAnswerEnd[i]]))
    print()