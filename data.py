import json, math
import numpy as np
import nltk
nltk.download('punkt')

from keras.preprocessing.sequence import pad_sequences


class Data:
    def __init__(self, config):
        self.batch_size = config.batch_size

        print('Preparing embedding matrix.')

        # load training data, parse, and split
        print('Loading in training data...')
        trainData = self.importMsmarco(config.train_path)
        self.tContext, self.tQuestion, self.tQuestionID, self.tAnswerBegin, self.tAnswerEnd, self.tAnswerText, self.maxLenTContext, self.maxLenTQuestion = self.splitMsmarcoDatasets(
            trainData)

        # load validation data, parse, and split
        print('Loading in validation data...')
        valData = self.importMsmarco(config.val_path)
        self.vContext, self.vQuestion, self.vQuestionID, self.vAnswerBegin, self.vAnswerEnd, self.vAnswerText, self.maxLenVContext, self.maxLenVQuestion = self.splitMsmarcoDatasets(
            valData)

        print('Building vocabulary...')
        # build a vocabular over all training and validation context paragraphs and question words
        vocab = self.buildVocab(self.tContext + self.tQuestion + self.vContext + self.vQuestion)

        # Reserve 0 for masking via pad_sequences
        config.vocab_size = len(vocab) + 1
        word_index = dict((c, i + 1) for i, c in enumerate(vocab))
        config.max_context_size = max(self.maxLenTContext, self.maxLenVContext)
        config.max_ques_size = max(self.maxLenTQuestion, self.maxLenVQuestion)

        # Note: Need to download and unzip Glove pre-train model files into same file as this script
        embeddings_index = self.loadGloveModel('./datasets/glove/glove.6B.' + str(config.dim_size) + 'd.txt')
        self.embeddings = self.createEmbeddingMatrix(embeddings_index, word_index)

        # vectorize training and validation datasets
        print('Begin vectorizing process...')

        # tX: training Context, tXq: training Question, tYBegin: training Answer Begin ptr,
        # tYEnd: training Answer End ptr
        self.tX, self.tXq, self.tYBegin, self.tYEnd = self.vectorizeData(self.tContext, self.tQuestion, self.tAnswerBegin, self.tAnswerEnd, word_index,
                                                     config.max_context_size, config.max_ques_size)
        self.vX, self.vXq, self.vYBegin, self.vYEnd = self.vectorizeData(self.vContext, self.vQuestion, self.vAnswerBegin, self.vAnswerEnd, word_index,
                                                     config.max_context_size, config.max_ques_size)
        print('Vectorizing process completed.')

        self.num_examples = len(self.tX)

    def get_all_data(self):
        return self.all_data

    def get_num_batches(self):
        return int(math.ceil(self.num_examples / self.batch_size))

    def get_train_batch(self):
        assert len(self.tX) == len(self.tXq)
        assert len(self.tX) == len(self.tYBegin)
        assert len(self.tX) == len(self.tYEnd)
        points = np.random.choice(len(self.tX), self.batch_size)
        tContext_batch = np.array(self.tContext)[points]
        tQuestion_batch = np.array(self.tQuestion)[points]
        tX_batch = np.array(self.tX)[points]
        tXq_batch = np.array(self.tXq)[points]
        tYBegin_batch = np.array(self.tYBegin)[points]
        tYEnd_batch = np.array(self.tYEnd)[points]

        return {'tContext': tContext_batch, 'tQuestion': tQuestion_batch,
                'tX': tX_batch, 'tXq': tXq_batch,
                'tYBegin': tYBegin_batch, 'tYEnd': tYEnd_batch}

    def get_val_batch(self):
        assert len(self.vX) == len(self.vXq)
        assert len(self.vX) == len(self.vYBegin)
        assert len(self.vX) == len(self.vYEnd)
        points = np.random.choice(len(self.vX), self.batch_size)
        vContext_batch = np.array(self.vContext)[points]
        vQuestion_batch = np.array(self.vQuestion)[points]
        vX_batch = np.array(self.vX)[points]
        vXq_batch = np.array(self.vXq)[points]
        vYBegin_batch = np.array(self.vYBegin)[points]
        vYEnd_batch = np.array(self.vYEnd)[points]

        return {'vContext': vContext_batch, 'vQuestion': vQuestion_batch,
                'vX': vX_batch, 'vXq': vXq_batch,
                'vYBegin': vYBegin_batch, 'vYEnd': vYEnd_batch}

    def loadGloveModel(self, gloveFile):
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

    def createEmbeddingMatrix(self, embeddings_index, word_index):
        dim_size = len(embeddings_index['a'])
        # +1 is for <PAD>
        embedding_matrix = np.zeros((len(word_index) + 1, dim_size))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def tokenize(self, sent):
        '''Return the tokens of a context including punctuation. Wrapper around nltk.word_tokenize to
           fix weird quotation marks.
        '''
        return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]

    def splitMsmarcoDatasets(self, f):
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

        # For now only pick out selected passages that have answers directly inside the passage
        for data in f['data']:
            for passage in data['passages']:
                if passage['is_selected'] == 0:
                    continue
                context = passage['passage_text']
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')
                contextTokenized = self.tokenize(context.lower())
                contextLength = len(contextTokenized)
                if contextLength > maxLenContext:
                    maxLenContext = contextLength

                question = data['query']
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = self.tokenize(question.lower())
                if len(questionTokenized) > maxLenQuestion:
                    maxLenQuestion = len(questionTokenized)

                question_id = data['query_id']

                answerFound = False

                for answer in data['answers']:
                    answerTokenized = self.tokenize(answer.lower())
                    answerBeginIndex, answerEndIndex = self.findAnswer(contextTokenized, answerTokenized)
                    if answerBeginIndex != None:
                        xContext.append(contextTokenized)
                        xQuestion.append(questionTokenized)
                        xQuestion_id.append(question_id)
                        xAnswerBegin.append(answerBeginIndex)
                        xAnswerEnd.append(answerEndIndex)
                        xAnswerText.append(answer)
                        answerFound = True
                        break

                if answerFound:
                    answerFound = False
                    break

        return xContext, xQuestion, xQuestion_id, xAnswerBegin, xAnswerEnd, xAnswerText, maxLenContext, maxLenQuestion

    def findAnswer(self, contextTokenized, answerTokenized):
        contextLen = len(contextTokenized)
        answerLen = len(answerTokenized)
        for i in range(contextLen - answerLen + 1):
            match = sum([1 for j, m in zip(contextTokenized[i:i + answerLen], answerTokenized) if j == m])
            if match == answerLen:
                return (i, i + answerLen)
        return (None, None)

    def vectorizeData(self, xContext, xQuestion, xAnswerBegin, xAnswerEnd, word_index, context_maxlen, question_maxlen):
        '''Converts context and question words to their respective index and pad context to max context length
           and question to max question length. *Convert answers to one-hot vectors of length max context.
        '''
        X = []
        Xq = []
        YBegin = []
        YEnd = []
        for i in range(len(xContext)):
            x = [word_index[w] for w in xContext[i]]
            xq = [word_index[w] for w in xQuestion[i]]
            # map the first and last words of answer span to one-hot representations
            y_Begin =  xAnswerBegin[i]
            y_End = xAnswerEnd[i] - 1
            X.append(x)
            Xq.append(xq)
            YBegin.append(y_Begin)
            YEnd.append(y_End)
        return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post'), YBegin, YEnd

    def saveAnswersForEval(self, referencesPath, candidatesPath, vContext, vQuestionID, predictedBegin, predictedEnd, trueBegin, trueEnd):
        rf = open(referencesPath, 'w', encoding='utf-8')
        cf = open(candidatesPath, 'w', encoding='utf-8')

        for i in range(len(vContext)):
            predictedAnswer = ' '.join(vContext[i][predictedBegin[i] : predictedEnd[i]])
            trueAnswer =' '.join(vContext[i][trueBegin[i] : trueEnd[i]])

            reference = {}
            candidate = {}
            reference['query_id'] = vQuestionID[i]
            reference['answers'] = [trueAnswer]

            candidate['query_id'] = vQuestionID[i]
            candidate['answers'] = [predictedAnswer]

            print(json.dumps(reference), file=rf)
            print(json.dumps(candidate), file=cf)

        rf.close()
        cf.close()

    def importMsmarco(self, json_file):
        data = {}
        data['data'] = []
        with open(json_file, encoding='utf-8') as f:
            data['data'] = [json.loads(line) for line in f]
        return data

    def buildVocab(self, sentences):
        '''Accepts a list of list of words. For example, a list of contexts or questions that are tokenized.
           Returns a sorted list of strings that comprise the vocabulary.
        '''
        vocab = set([word for words in sentences for word in words])
        vocab = sorted(vocab)
        return vocab