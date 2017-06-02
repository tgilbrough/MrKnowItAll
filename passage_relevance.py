import json
import collections 
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from matplotlib import pyplot as plt


def load_samples(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def load_passages(path):
    print('Loading passages from {}...'.format(path))

    samples = []

    for sample in load_samples(path):
        passages = []
        selected = []
        s = {}
        for passage in sample['passages']:
            passages.append(passage['passage_text'].lower())
            selected.append(passage['is_selected'])
        s['query'] = sample['query'].lower()
        s['selected'] = selected
        s['passages'] = passages
        samples.append(s)

    print('Loaded {} samples.'.format(len(samples)))

    return samples

def get_rank(t):
    tfidf = TfidfVectorizer(binary=True).fit_transform([t['query']] + t['passages'])
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()[1:]
    related_docs_indices = cosine_similarities.argsort()[::-1]
    try:
        selected_index = t['selected'].index(1)
        return related_docs_indices.tolist().index(selected_index)
    except ValueError:
        return None


def get_accuracy(passages):
    rank_counter = collections.Counter(get_rank(t) for t in passages)
    counts = [count for element, count in rank_counter.items()
              if element is not None]
    cumsum = np.cumsum(counts)
    percents = [x / cumsum[-1] for x in cumsum]
    return percents


def main():
    for question_type in ['description', 'numeric', 'entity', 'location', 'person']:
        path = 'datasets/msmarco/train/{}.json'.format(question_type)
        passages = load_passages(path)
        accuracy = get_accuracy(passages)
        plt.plot(range(1, len(accuracy) + 1), accuracy, label=question_type)

    plt.ylabel('Percent of Questions where Selected Passage is Included')
    plt.xlabel('Number of Passages')
    plt.legend(loc='lower right')
    plt.show()

main()


def a():
    train = load_passages('datasets/msmarco/train/location.json')

    for t in train[:0]:
        tfidf = TfidfVectorizer().fit_transform([t['query']] + t['passages'])
        cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()[1:]
        related_docs_indices = cosine_similarities.argsort()[::-1]

        print('Cosine Similarities:', cosine_similarities)
        print('Ranked indexes:', related_docs_indices)
        print('Selected:', t['selected'])
        print('Query:', t['query'] + '?')
        print()
        print('Passages:')
        for i in range(len(t['passages'])):
            print(i, ':', t['passages'][i])
            print()
