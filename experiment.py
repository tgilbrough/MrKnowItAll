import argparse
import subprocess


QUESTION_TYPES = ['description', 'entity', 'location', 'numeric', 'person']
MODELS = ['baseline', 'attention']

def get_permutations(all_options):
    """ {'a': [1,2], 'b': [3, 4], 'c': 5} ->
        {'a': 1, 'c': 5, 'b': 3}
        {'a': 2, 'c': 5, 'b': 3}
        {'a': 1, 'c': 5, 'b': 4}
        {'a': 2, 'c': 5, 'b': 4}
    """

    if len(all_options) < 1:
        yield {}
    else:
        name, options = all_options.popitem()

        for child_options in get_permutations(all_options):
            if not isinstance(options, list):
                options = [options]

            for option in options:
                a = child_options.copy()
                a[name] = option
                yield a


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_type', '-q', nargs='+', default='all',
                        choices=QUESTION_TYPES + ['all'])
    parser.add_argument('--keep_prob', '-kp', type=float, default=0.7, nargs='+')
    parser.add_argument('--hidden_size', '-hs', type=int, default=100, nargs='+')
    parser.add_argument('--emb_size', '-es', type=int, default=50, nargs='+') # this could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    parser.add_argument('--epochs', '-e', type=int, default=50, nargs='+')
    parser.add_argument('--batch_size', '-bs', type=int, default=64, nargs='+')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, nargs='+')
    parser.add_argument('--num_threads', '-t', type=int, default=4, nargs='+')
    parser.add_argument('--model', '-m', default='baseline', nargs='+', choices=MODELS + ['all'])

    args = vars(parser.parse_args())
    if 'all' in args['question_type']:
        args['question_type'] = QUESTION_TYPES
    if 'all' in args['model']:
        args['model'] = MODELS

    return args


def format_args(args):
    for name, value in args.items():
        yield '--{}'.format(name)
        yield str(value)


def get_experiment_name(options, non_default_args):
    option_summary = '-'.join('{}={}'.format(key, value)
                              for key, value in sorted(options.items())
                              if key != 'model' and key in non_default_args)
    return '{}-{}'.format(options['model'], option_summary)


args = get_args()
non_default_args = [name for name, value in args.items()
                    if isinstance(value, list)]

for options in get_permutations(args):
    print(options)
    options['tensorboard_name'] = get_experiment_name(options, non_default_args)
    args = ['python', 'main.py'] + list(format_args(options))
    subprocess.call(args)

