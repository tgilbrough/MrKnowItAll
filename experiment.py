import argparse
import subprocess


QUESTION_TYPES = ['description', 'entity', 'location', 'numeric', 'person']


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
    parser.add_argument('name')
    parser.add_argument('--question_type', '-q', nargs='+', default='all',
                        choices=QUESTION_TYPES + ['all'])
    parser.add_argument('--batch_size', '-bs', nargs='+', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=100)

    args = vars(parser.parse_args())
    if 'all' in args['question_type']:
        args['question_type'] = QUESTION_TYPES

    return args


def format_args(args):
    for name, value in args.items():
        yield '--{}'.format(name)
        yield str(value)


def get_experiment_name(options):
    option_summary = '-'.join('{}:{}'.format(key, value)
                              for key, value in sorted(options.items())
                              if key != 'name')
    return '{}-{}'.format(options['name'], option_summary)


for options in get_permutations(get_args()):
    print(options)
    options['tensorboard_name'] = get_experiment_name(options)
    del(options['name'])
    args = ['python3', 'main.py'] + list(format_args(options))
    subprocess.run(args)

