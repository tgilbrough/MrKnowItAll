

import nltk
nltk.download('punkt', quiet=True)

import argparse
import json


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


def mark(string, to_mark, css_class):
    start = string.index(to_mark)
    end = start + len(to_mark)
    marked = insert(string, '</mark>', end)
    marked = insert(marked, '<mark class="{}">'.format(css_class), start)
    return marked


def format_candidate(passage, candidate):
    return ('<td class="candidate">{}</td>'
            .format(mark(passage, candidate, 'candidate')))


def print_query(query, passage, reference, candidates):
    colspan = len(candidates) + 1

    print('''
        <tr>
          <td class="query" colspan="{}">{}?</td>
        </tr>
        <tr class="passage">
          <td class="reference">{}</td>
          {}
        </tr>
        <tr>
          <td class="padding" colspan="{}"></td>
        </tr>
    '''.format(colspan,
               query['query'],
               mark(passage, reference, 'reference'),
               '\n'.join(format_candidate(passage, candidate)
                         for candidate in candidates),
               colspan))


def format_parameter(parameter):
    return '<span class="parameter">{}</span>'.format(parameter)


def format_candidate_header(path):
    elements = path.split('-')
    model_name = elements[0].split('/')[-1]
    return '''
      <th>
        <span class="model-name">{}</span>
        {}
      </th>
      '''.format(model_name,
                 '\n'.join(format_parameter(param) for param in elements[1:]))

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('reference_path')
parser.add_argument('candidate_paths', nargs='+')
args = parser.parse_args()

query_generator = load_json_lines(args.data_path)
candidate_generators = [load_json_lines(candidate_path)
                        for candidate_path in args.candidate_paths]
reference_generator = load_json_lines(args.reference_path)

print('''
<html>
  <head>
    <link rel="stylesheet" href="main.css" />
  </head>
  <body>
    <div>
      <table>
        <tr>
          <th>Reference</th>
          {}
        </tr>
        <tr>
          <td class="padding" colspan="2"></td>
        </tr>
'''.format('\n'.join(format_candidate_header(candidate)
                     for candidate in args.candidate_paths)))


for reference in reference_generator:
    candidates = [next(candidate_generator)
                  for candidate_generator in candidate_generators]

    if any(reference['query_id'] != candidate['query_id']
           for candidate in candidates):
        print('candidate and reference query id do not match')

    query = next(query for query in query_generator
                 if query['query_id'] == reference['query_id'])

    passage = next(get_text(passage) for passage in query['passages']
                   if passage['is_selected'] == 1 and
                   reference['answers'][0] in get_text(passage))

    print_query(query, passage, reference['answers'][0],
                [candidate['answers'][0] for candidate in candidates])


print('''
      </table>
    </div>
  </body>
</html>
''')
