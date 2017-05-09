import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

train = load_passages('datasets/msmarco/train/location.json')
for t in train[:10]:
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