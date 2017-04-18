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
        tContext, tQuestion, tQuestionID, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = self.splitMsmarcoDatasets(
            trainData)

        # load validation data, parse, and split
        print('Loading in validation data...')
        valData = self.importMsmarco(config.val_path)
        vContext, vQuestion, vQuestionID, vAnswerBegin, vAnswerEnd, vAnswerText, maxLenVContext, maxLenVQuestion = self.splitMsmarcoDatasets(
            valData)

        print('Building vocabulary...')
        # build a vocabular over all training and validation context paragraphs and question words
        vocab = self.buildVocab(tContext + tQuestion + vContext + vQuestion)

        # Reserve 0 for masking via pad_sequences
        config.vocab_size = len(vocab) + 1
        word_index = dict((c, i + 1) for i, c in enumerate(vocab))
        config.max_context_size = max(maxLenTContext, maxLenVContext)
        config.max_ques_size = max(maxLenTQuestion, maxLenVQuestion)

        # Note: Need to download and unzip Glove pre-train model files into same file as this script
        embeddings_index = self.loadGloveModel('./datasets/glove/glove.6B.' + str(config.dim_size) + 'd.txt')
        embeddings = self.createEmbeddingMatrix(embeddings_index, word_index)

        # vectorize training and validation datasets
        print('Begin vectoring process...')

        # tX: training Context, tXq: training Question, tYBegin: training Answer Begin ptr,
        # tYEnd: training Answer End ptr
        tX, tXq, tYBegin, tYEnd = self.vectorizeData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index,
                                                     config.max_context_size, config.max_ques_size)
        vX, vXq, vYBegin, vYEnd = self.vectorizeData(vContext, vQuestion, vAnswerBegin, vAnswerEnd, word_index,
                                                     config.max_context_size, config.max_ques_size)
        print('Vectoring process completed.')

        self.num_examples = len(tX)
        self.all_data = {'tX': tX, 'tXq': tXq, 'tYBegin': tYBegin, 'tYEnd': tYEnd,
                         'vX': vX, 'vXq': vXq, 'vYBegin': vYBegin, 'vYEnd': vYEnd}

    def get_all_data(self):
        return self.all_data

    def get_num_batches(self):
        return int(math.ceil(self.num_examples / self.batch_size))

    def get_batch(self):
        assert len(self.all_data['tX']) == len(self.all_data['tXq'])
        assert len(self.all_data['tX']) == len(self.all_data['tYBegin'])
        assert len(self.all_data['tX']) == len(self.all_data['tYEnd'])
        points = np.random.choice(len(self.all_data['tX']), self.batch_size)
        tX_batch = np.array(self.all_data['tX'])[points]
        tXq_batch = np.array(self.all_data['tX'])[points]
        tYBegin_batch = np.array(self.all_data['tX'])[points]
        tYEnd_batch = np.array(self.all_data['tX'])[points]

        return {'tX_batch': tX_batch, 'tXq_batch': tXq_batch,
                'tYBegin_batch': tYBegin_batch, 'tYEnd_batch': tYEnd_batch}




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

    def splitSquadDatasets(self, f):
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
                contextTokenized = self.tokenize(context.lower())
                contextLength = len(contextTokenized)
                if contextLength > maxLenContext:
                    maxLenContext = contextLength
                qas = paragraph['qas']
                for qa in qas:
                    question = qa['question']
                    question = question.replace("''", '" ')
                    question = question.replace("``", '" ')
                    questionTokenized = self.tokenize(question.lower())
                    if len(questionTokenized) > maxLenQuestion:
                        maxLenQuestion = len(questionTokenized)
                    question_id = qa['id']
                    answers = qa['answers']
                    for answer in answers:
                        answerText = answer['text']
                        answerTokenized = self.tokenize(answerText.lower())
                        # find indices of beginning/ending words of answer span among tokenized context
                        contextToAnswerFirstWord = context1[:answer['answer_start'] + len(answerTokenized[0])]
                        answerBeginIndex = len(self.tokenize(contextToAnswerFirstWord.lower())) - 1
                        answerEndIndex = answerBeginIndex + len(answerTokenized)

                        xContext.append(contextTokenized)
                        xQuestion.append(questionTokenized)
                        xQuestion_id.append(str(question_id))
                        xAnswerBegin.append(answerBeginIndex)
                        xAnswerEnd.append(answerEndIndex)
                        xAnswerText.append(answerText)
        return xContext, xQuestion, xQuestion_id, xAnswerBegin, xAnswerEnd, xAnswerText, maxLenContext, maxLenQuestion

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

    def importSquad(self, json_file):
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        return data

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