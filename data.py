import json, math
import numpy as np
import os
import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Data:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.keep_prob = config.keep_prob
        self.valBatchNum = 0
        self.testBatchNum = 0

        print('Preparing embedding matrix.')

        # load training data, parse, and split
        print('Loading in training data...')
        trainData = self.importMsmarco(config.train_path)
        self.tContext, self.tQuestion, self.tQuestionID, self.tAnswerBegin, self.tAnswerEnd, self.tAnswerText, \
            self.maxLenTContext, self.maxLenTQuestion = self.splitMsmarcoDatasets(trainData)

        # load validation data, parse, and split
        print('Loading in validation data...')
        valData = self.importMsmarco(config.val_path)
        self.vContext, self.vQuestion, self.vQuestionID, self.vAnswerBegin, self.vAnswerEnd, self.vAnswerText, \
            self.maxLenVContext, self.maxLenVQuestion = self.splitMsmarcoDatasets(valData)

        # load test data, parse, and split
        print('Loading in testing data...')
        testData = self.importMsmarco(config.test_path)
        self.teContext, self.teQuestion, self.teQuestionID, self.teUrls, \
            self.maxLenTeContext, self.maxLenTeQuestion = self.splitMsmarcoDatasetsTest(testData)

        print('Building vocabulary...')
        # build a vocabulary over all training and validation context paragraphs and question words
        vocab = self.buildVocab(self.tContext + self.tQuestion + self.vContext + self.vQuestion 
                                + [t for p in self.teContext for t in p] + self.teQuestion)
        self.vocab = vocab

        # Reserve 0 for masking via pad_sequences
        self.vocab_size = len(vocab) + 1
        word_index = dict((c, i + 1) for i, c in enumerate(vocab))
        self.max_context_size = max([self.maxLenTContext, self.maxLenVContext, self.maxLenTeContext])
        self.max_ques_size = max([self.maxLenTQuestion, self.maxLenVQuestion, self.maxLenTeQuestion])

        # Note: Need to download and unzip Glove pre-train model files into same file as this script
        embeddings_index = self.loadGloveModel('./datasets/glove/glove.6B.' + str(config.emb_size) + 'd.txt')
        self.embeddings = self.createEmbeddingMatrix(embeddings_index, word_index)

        # Calculating passage relevance weights
        self.tePassWeights = self.passageRevelevance(self.teContext, self.teQuestion)

        # vectorize training and validation datasets
        print('Begin vectorizing process...')

        # tX: training Context, tXq: training Question, tYBegin: training Answer Begin ptr,
        # tYEnd: training Answer End ptr
        self.tX, self.tXq, self.tYBegin, self.tYEnd = self.vectorizeData(self.tContext, self.tQuestion,
                                                     self.tAnswerBegin, self.tAnswerEnd, word_index,
                                                     self.max_context_size, self.max_ques_size)
        self.vX, self.vXq, self.vYBegin, self.vYEnd = self.vectorizeData(self.vContext, self.vQuestion,
                                                     self.vAnswerBegin, self.vAnswerEnd, word_index,
                                                     self.max_context_size, self.max_ques_size)
        self.teX, self.teXq = self.vectorizeDataTest(self.teContext, self.teQuestion, word_index,
                                                     self.max_context_size, self.max_ques_size)

        print('Vectorizing process completed.')

        self.convertToNumpy()

    def getAllData(self):
        return self.all_data

    def convertToNumpy(self):
        self.tX = np.array(self.tX)
        self.tXq = np.array(self.tXq)
        self.tYBegin = np.array(self.tYBegin)
        self.tYEnd = np.array(self.tYEnd)

        self.vX = np.array(self.vX)
        self.vXq = np.array(self.vXq)
        self.vYBegin = np.array(self.vYBegin)
        self.vYEnd = np.array(self.vYEnd)
        self.vContext = np.array(self.vContext, dtype=object)

        self.teX = np.array(self.teX)
        self.teXq = np.array(self.teXq)
        self.teContext = np.array(self.teContext, dtype=object)
        self.tePassWeights = np.array(self.tePassWeights)

    def getNumTrainBatches(self):
        return int(math.ceil(len(self.tX) / self.batch_size))

    def getNumValBatches(self):
        return int(math.ceil(len(self.vX) / self.batch_size))

    def getNumTestBatches(self):
        return int(math.ceil(len(self.teX) / self.batch_size))

    def getRandomTrainBatch(self):
        points = np.random.choice(len(self.tX), self.batch_size)

        tX_batch = self.tX[points]
        tXq_batch = self.tXq[points]
        tYBegin_batch = self.tYBegin[points]
        tYEnd_batch = self.tYEnd[points]

        return {'tX': tX_batch, 'tXq': tXq_batch,
                'tYBegin': tYBegin_batch, 'tYEnd': tYEnd_batch}

    def getRandomValBatch(self):
        points = np.random.choice(len(self.vX), self.batch_size)

        vX_batch = self.vX[points]
        vXq_batch = self.vXq[points]
        vYBegin_batch = self.vYBegin[points]
        vYEnd_batch = self.vYEnd[points]

        return {'vX': vX_batch, 'vXq': vXq_batch,
                'vYBegin': vYBegin_batch, 'vYEnd': vYEnd_batch}

    def getValBatch(self):
        start = self.valBatchNum * self.batch_size
        end = min(len(self.vX), (self.valBatchNum + 1) * self.batch_size)
        points = np.arange(start, end)

        vContext_batch = self.vContext[points]
        vQuestionID_batch = self.vQuestionID[points]
        vX_batch = self.vX[points]
        vXq_batch = self.vXq[points]
        vYBegin_batch = self.vYBegin[points]
        vYEnd_batch = self.vYEnd[points]

        self.valBatchNum += 1

        if self.valBatchNum >= self.getNumValBatches():
            self.valBatchNum = 0

        return {'vContext': vContext_batch, 'vQuestionID': vQuestionID_batch,
                'vX': vX_batch, 'vXq': vXq_batch,
                'vYBegin': vYBegin_batch, 'vYEnd': vYEnd_batch}

    def getTestBatch(self):
        start = self.testBatchNum * self.batch_size
        end = min(len(self.teX), (self.testBatchNum + 1) * self.batch_size)
        points = np.arange(start, end)

        teContext_batch = self.teContext[points]
        teQuestionID_batch = self.teQuestionID[points]
        teX_batch = self.teX[points]
        teXq_batch = self.teXq[points]
        teUrl_batch = self.teUrls[points]
        teXPassWeights_batch = self.tePassWeights[points]

        self.testBatchNum += 1

        if self.testBatchNum >= self.getNumTestBatches():
            self.testBatchNum = 0

        return {'teContext': teContext_batch, 'teQuestionID': teQuestionID_batch,
                'teX': teX_batch, 'teXq': teXq_batch, 'teUrl': teUrl_batch,
                'teXPassWeights': teXPassWeights_batch}

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

    def join(self, sent):
        def join_punctuation(seq, characters='.,;?!'):
            characters = set(characters)
            seq = iter(seq)
            current = next(seq)

            for nxt in seq:
                if nxt in characters:
                    current += nxt
                else:
                    yield current
                    current = nxt

            yield current
        return ' '.join(join_punctuation(sent))



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

        seen_query_ids = set()

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

    def splitMsmarcoDatasetsTest(self, f):
        '''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices,
           and keep track of max context and question lengths.
        '''
        xContext = [] # list of list of contexts paragraphs
        xQuestion = [] # list of questions
        xQuestionID = [] # list of question id
        xUrls = [] # list of list of urls associated with the passages
        maxLenContext = 0
        maxLenQuestion = 0

        for data in f['data']:
            passages = []
            urls = []

            for passage in data['passages']:
                context = passage['passage_text']
                contextTokenized = self.tokenize(context.lower())

                passages.append(contextTokenized)
                urls.append(passage['url'])

                contextLength = len(contextTokenized)
                if contextLength > maxLenContext:
                    maxLenContext = contextLength

            question = data['query']
            questionTokenized = self.tokenize(question.lower())
            if len(questionTokenized) > maxLenQuestion:
                maxLenQuestion = len(questionTokenized)

            questionID = data['query_id']

            xContext.append(passages)
            xQuestion.append(questionTokenized)
            xQuestionID.append(questionID)
            xUrls(urls)

        return xContext, xQuestion, xQuestionID, xUrls, maxLenContext, maxLenQuestion


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
        return self.pad_sequences(X, context_maxlen), self.pad_sequences(Xq, question_maxlen), YBegin, YEnd

    def vectorizeDataTest(self, xContext, xQuestion, word_index, context_maxlen, question_maxlen):
        '''Converts context and question words to their respective index and pad context to max context length
           and question to max question length. *Convert answers to one-hot vectors of length max context.
        '''
        X = []
        Xq = []
        for i in range(len(xContext)):
            xs = [[word_index[w] for w in p] for p in xContext[i]]
            xq = [word_index[w] for w in xQuestion[i]]
            # map the first and last words of answer span to one-hot representations
            X.append(self.pad_sequences(xs, context_maxlen))
            Xq.append(xq)
        return X, self.pad_sequences(Xq, question_maxlen)

    def pad_sequences(self, X, maxlen):
        for context in X:
            for i in range(maxlen - len(context)):
                context.append(0)
        return X

    def passageRevelevance(self, xContext, xQuestion):
        cs = []
        for i in range(len(xContext)):
            passages = [' '.join(p) for p in xContext[i]]

            tfidf = TfidfVectorizer().fit_transform([' '.join(xQuestion[i])] + passages)
            cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()[1:]
            
            # Normalize
            # cosine_similarities = [math.exp(x) for x in cosine_similarities]
            sum_cs = sum(cosine_similarities)
            cosine_similarities = [x / sum_cs for x in cosine_similarities]

            cs.append(cosine_similarities)
        
        return cs

    def saveAnswersForEvalVal(self, questionType, candidateName, vContext, vQuestionID, predictedBegin, predictedEnd, trueBegin, trueEnd):
        ref_fn = './references/' + questionType + '.json'
        can_fn = './candidates/' + candidateName + '.json'

        rf = open(ref_fn, 'w', encoding='utf-8')
        cf = open(can_fn, 'w', encoding='utf-8')

        for i in range(len(vContext)):
            predictedAnswer = ' '.join(vContext[i][predictedBegin[i] : predictedEnd[i] + 1])
            trueAnswer = ' '.join(vContext[i][trueBegin[i] : trueEnd[i] + 1])

            reference = {}
            candidate = {}
            reference['query_id'] = vQuestionID[i]
            reference['answers'] = [trueAnswer]

            candidate['query_id'] = vQuestionID[i]
            candidate['answers'] = [predictedAnswer]

            print(json.dumps(reference, ensure_ascii=False), file=rf)
            print(json.dumps(candidate, ensure_ascii=False), file=cf)

        rf.close()
        cf.close()

    def saveAnswersForEvalTest(self, questionType, candidateName, teContext, teQuestionID, predictedBegin, predictedEnd):
        can_fn = './candidates_multi/' + candidateName + '.json'

        cf = open(can_fn, 'w', encoding='utf-8')

        for i in range(len(teContext)):
            predictedAnswer = ' '.join(teContext[i][predictedBegin[i] : predictedEnd[i] + 1])

            candidate = {}
            candidate['query_id'] = teQuestionID[i]
            candidate['answers'] = [predictedAnswer]

            print(json.dumps(candidate, ensure_ascii=False), file=cf)

        cf.close()

    def saveAnswersForEvalTestDemo(self, questionType, candidateName, teContext, teQuestionID, teUrl, predictedBegin, predictedEnd, passageWeights, logitsStart, logitsEnd):
        can_fn = './candidates_multi/' + candidateName + '.json'

        cf = open(can_fn, 'w', encoding='utf-8')

        for i in range(len(teContext)):
            predictedAnswer = ' '.join(teContext[i][predictedBegin[i] : predictedEnd[i] + 1])

            candidate = {}
            candidate['query_id'] = teQuestionID[i]
            candidate['answers'] = [predictedAnswer]
            candidate['relevance'] = passageWeights[i]
            candidate['logits_start'] = logitsStart[i]
            candidate['logits_end'] = logitsEnd[i]

            print(json.dumps(candidate, ensure_ascii=False), file=cf)

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
