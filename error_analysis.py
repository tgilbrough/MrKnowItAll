

import nltk
nltk.download('punkt')

import json

DATA_PATH = 'datasets/msmarco/dev/location.json'
CANDIDATE_PATH = 'eval/candidates.json'
REFERENCE_PATH = 'eval/references.json'


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
    marked = insert(passage, '</mark>', start)
    marked = insert(passage, '<mark class="{}">'.format(css_class), end)
    return marked


def print_html(query, passage, candidate, reference):
    candidate_start = passage.index(candidate)
    candidate_end = candidate_start + len(candidate)
    reference_start = passage.index(reference)
    reference_end = reference_start + len(reference)

#     if (candidate_start < reference_start and candidate_end < reference_start):
#         passage = mark(passage, reference_start, reference_end, 'reference')
#         passage = mark(passage, candidate_start, candidate_end, 'candidate')
#     elif (reference_start < candidate_start and reference_end < candidate_start):
#         passage = mark(passage, candidate_start, candidate_end, 'candidate')
#         passage = mark(passage, reference_start, reference_end, 'reference')

    def get_class(index):
        in_candidate = index >= candidate_start and index <= candidate_end
        in_reference = index >= reference_start and index <= reference_end
        if (in_candidate and in_reference):
            return 'match'
        if (in_candidate):
            return 'candidate'
        if (in_reference):
            return 'reference'
        return ''

    passage = ' '.join('<span class="{}">{}</span>'
                       .format(get_class(i), passage[i])
                       for i in range(len(passage)))

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
      div {
        font-size: 0;
      }

      span {
        font-size: 12px;
      }

      span.reference {
        background-color: green;
      }

      span.match {
        background-color: DarkGreen;
      }

      span.candidate {
        background-color: red;
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
               reference['answers'][0],
               candidate['answers'][0])


print('''
    </body>
</html>
''')
