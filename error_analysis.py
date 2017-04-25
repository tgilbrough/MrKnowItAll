

import nltk
nltk.download('punkt')

import json

DATA_PATH = 'datasets/msmarco/dev/location.json'
CANDIDATE_PATH = 'candidates/baseline-batch_size=1024-epochs=50-hidden_size=50-keep_prob=0.5-learning_rate=0.01-question_type=location.json'
REFERENCE_PATH = 'references/location.json'


def load_json_lines(path):
    with open(path) as json_file:
        for line in json_file:
            yield json.loads(line)


def tokenize(string):
    return (token.replace("``", '"').replace("''", '"')
            for token in nltk.word_tokenize(string))


def get_text(passage):
    return ' '.join(tokenize(passage['passage_text'].lower()))


def insert(string, to_insert, index):
    return string[:index] + to_insert + string[index:]


def mark(string, start, end, css_class):
    marked = insert(string, '</mark>', end)
    marked = insert(marked, '<mark class="{}">'.format(css_class), start)
    return marked


def print_html(query, passage, candidate, reference):
    candidate_start = passage.index(candidate)
    candidate_end = candidate_start + len(candidate)
    reference_start = passage.index(reference)
    reference_end = reference_start + len(reference)

    print(candidate_start)
    print(candidate_end)
    print(reference_start)
    print(reference_end)
    print(len(passage))

    if (candidate_start == reference_start and candidate_end == reference_end):
        passage = mark(passage, candidate_start, candidate_end, 'match')
    elif (candidate_start == reference_start and candidate_end < reference_end):
        passage = mark(passage, candidate_end, reference_end, 'reference')
        passage = mark(passage, candidate_start, candidate_end, 'match')
    elif (candidate_start == reference_start and candidate_end > reference_end):
        passage = mark(passage, reference_end, candidate_end, 'candidate')
        passage = mark(passage, candidate_start, reference_end, 'match')
    elif (candidate_start < reference_start and candidate_end > reference_end):
        passage = mark(passage, reference_end, candidate_end, 'candidate')
        passage = mark(passage, reference_start, reference_end, 'match')
        passage = mark(passage, candidate_start, reference_start, 'candidate')
    elif (reference_start < candidate_start and reference_end > candidate_end):
        passage = mark(passage, candidate_end, reference_end, 'reference')
        passage = mark(passage, candidate_start, candidate_end, 'match')
        passage = mark(passage, reference_start, candidate_start, 'reference')
    elif (candidate_start <= reference_start):
        if (candidate_end < reference_start):
            passage = mark(passage, reference_start, reference_end, 'reference')
            passage = mark(passage, candidate_start, candidate_end, 'candidate')
        else:
            passage = mark(passage, candidate_end, reference_end, 'reference')
            passage = mark(passage, reference_start, candidate_end, 'match')
            passage = mark(passage, candidate_start, reference_start, 'candidate')
    elif (reference_start <= candidate_start):
        if (reference_end < candidate_start):
            passage = mark(passage, candidate_start, candidate_end, 'candidate')
            passage = mark(passage, reference_start, reference_end, 'reference')
        else:
            passage = mark(passage, reference_end, candidate_end, 'candidate')
            passage = mark(passage, candidate_start, reference_end, 'match')
            passage = mark(passage, reference_start, candidate_start, 'reference')

    print('''
        <div>
            <p class='query'>{}</p>
            <p class='passage'>{}</p>
        </div>
    '''.format(query['query'], passage))


query_generator = load_json_lines(DATA_PATH)
candidate_generator = load_json_lines(CANDIDATE_PATH)
reference_generator = load_json_lines(REFERENCE_PATH)

print('''
<html>
    <head>
        <style>

      mark.reference {
        background-color: #629bf7;
      }

      mark.match {
        background-color: #62f77e;
      }

      mark.candidate {
        background-color: #f76262;
      }

        </style>
    </head>
    <body>
''')


for reference in reference_generator:
    candidate = next(candidate_generator)

    if reference['query_id'] != candidate['query_id']:
        print('candidate and reference query id do not match')

    query = next(query for query in query_generator
                 if query['query_id'] == reference['query_id'])

    passage = next(get_text(passage) for passage in query['passages']
                   if passage['is_selected'] == 1 and
                   reference['answers'][0] in get_text(passage))

    print_html(query, passage,
               candidate['answers'][0],
               reference['answers'][0])


print('''
    </body>
</html>
''')
