

import nltk
nltk.download('punkt', quiet=True)

import argparse
import json


def load_json_lines(path):
    with open(path) as json_file:
        for line in json_file:
            yield json.loads(line)


def json_lookup(path):
    return {row['query_id']: row for row in load_json_lines(path)}


def tokenize(string):
    return (token.replace("``", '"').replace("''", '"')
            for token in nltk.word_tokenize(string))


def clean(text):
    return ' '.join(tokenize(text.lower()))


def get_text(passage):
    return clean(passage['passage_text'])


def insert(string, to_insert, index):
    return string[:index] + to_insert + string[index:]


def newline_indexes(passages):
    index = 0

    for passage in passages[:-1]:
        index += len(passage)
        yield index


def mark(string, to_mark, css_class, newlines):
    start = string.index(to_mark)
    end = start + len(to_mark)
    end_mark = '</mark>'
    marked = insert(string, end_mark, end)
    start_mark = '<mark class="{}">'.format(css_class)
    marked = insert(marked, start_mark, start)

    n_mark = '<hr>'
    for i, n_i in enumerate(newlines[::-1]):
        m_i = n_i + (len(newlines) - i)
        if n_i > end:
            m_i += len(end_mark)
        if n_i > start:
            m_i += len(start_mark)
        marked = insert(marked, n_mark, m_i)

    return marked


def format_candidate(passage, candidate, newlines):
    return ('<td class="candidate">{}</td>'
            .format(mark(passage, candidate, 'candidate', newlines)))


def print_query(query, reference, candidates):
    colspan = len(candidates) + 1

    passages = [get_text(p) for p in query['passages']]
    newlines = list(newline_indexes(passages))
    concat_passage = ' '.join(passages)

    try:
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
                   mark(concat_passage, reference, 'reference', newlines),
                   '\n'.join(format_candidate(concat_passage, candidate, newlines)
                             for candidate in candidates),
                   colspan))
    except ValueError:
        pass


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

query_lookup = json_lookup(args.data_path)
candidate_lookups = [json_lookup(c) for c in args.candidate_paths]
reference_lookup = json_lookup(args.reference_path)

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


for query_id, reference in reference_lookup.items():
    candidates = [c.get(query_id) for c in candidate_lookups]
    query = query_lookup.get(query_id)

    if any(c is None for c in candidates) or query is None:
        continue

    print_query(query,
                clean(reference['answers'][0]),
                [clean(candidate['answers'][0]) for candidate in candidates])

print('''
      </table>
    </div>
  </body>
</html>
''')
