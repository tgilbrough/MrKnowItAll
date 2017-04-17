import json
import numpy as np
import nltk
nltk.download('punkt')

from keras.preprocessing.sequence import pad_sequences

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
    '''Return the tokens of a context including punctuation. Wrapper around nltk.word_tokenize to
       fix weird quotation marks.
    '''
    return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]

def splitSquadDatasets(f):
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
                    answerEndIndex = answerBeginIndex + len(answerTokenized)

                    xContext.append(contextTokenized)
                    xQuestion.append(questionTokenized)
                    xQuestion_id.append(str(question_id))
                    xAnswerBegin.append(answerBeginIndex)
                    xAnswerEnd.append(answerEndIndex)
                    xAnswerText.append(answerText)
    return xContext, xQuestion, xQuestion_id, xAnswerBegin, xAnswerEnd, xAnswerText, maxLenContext, maxLenQuestion

def splitMsmarcoDatasets(f):
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
            contextTokenized = tokenize(context.lower())
            contextLength = len(contextTokenized)
            if contextLength > maxLenContext:
                maxLenContext = contextLength

            question = data['query']
            question = question.replace("''", '" ')
            question = question.replace("``", '" ')
            questionTokenized = tokenize(question.lower())
            if len(questionTokenized) > maxLenQuestion:
                maxLenQuestion = len(questionTokenized)

            question_id = data['query_id']
            
            answerFound = False

            for answer in data['answers']:
                answerTokenized = tokenize(answer.lower())
                answerBeginIndex, answerEndIndex = findAnswer(contextTokenized, answerTokenized)
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

def findAnswer(contextTokenized, answerTokenized):
    contextLen = len(contextTokenized)
    answerLen = len(answerTokenized)
    for i in range(contextLen - answerLen + 1):
        match = sum([1 for j, m in zip(contextTokenized[i:i + answerLen], answerTokenized) if j == m])
        if match == answerLen:
            return (i, i + answerLen)
    return (None, None)

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
        y_End[xAnswerEnd[i] - 1] = 1
        X.append(x)
        Xq.append(xq)
        YBegin.append(y_Begin)
        YEnd.append(y_End)
    return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post'), pad_sequences(YBegin, maxlen=context_maxlen, padding='post'), pad_sequences(YEnd, maxlen=context_maxlen, padding='post')

def saveAnswersForEval(referencesPath, candidatesPath, vContext, vQuestionID, predictedBegin, predictedEnd, trueBegin, trueEnd):
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

def importSquad(json_file):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    return data

def importMsmarco(json_file):
    data = {}
    data['data'] = []
    with open(json_file, encoding='utf-8') as f:
        data['data'] = [json.loads(line) for line in f]
    return data

def buildVocab(sentences):
    '''Accepts a list of list of words. For example, a list of contexts or questions that are tokenized.
       Returns a sorted list of strings that comprise the vocabulary.
    '''
    vocab = set([word for words in sentences for word in words])
    vocab = sorted(vocab)
    return vocab