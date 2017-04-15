
import nltk
nltk.download('punkt')


class Glove:
    def __init__(self, path='data/glove/glove.6B.50d.txt'):
        "Load GloVe embeddings from the given file."

        print('Loading GloVe embeddings from {}...'.format(path))

        with open(path, encoding='utf-8') as glove_file:
            self.index = {}

            for line in glove_file:
                values = line.split()
                self.index[values[0]] = [float(value) for value in values[1:]]

            self.dim = len(next(iter(self.index.values())))

        print('  Loaded {} {}-dimensional vectors.'
              .format(len(self.index), self.dim))

    def paragraph(self, paragraph):
        "Embed a paragraph by taking the average of its words."

        tokens = nltk.word_tokenize(paragraph)
        vectors = [self.index[token]
                   for token in tokens
                   if token in self.index]
        return [sum(vector[i] for vector in vectors) / len(vectors)
                for i in range(self.dim)]
