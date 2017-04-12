import argparse
import os
import data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=os.path.join('data', 'train'))
    parser.add_argument('--glove_path', default=os.path.join('data', 'glove', 'glove.6B.50d.txt'))
    parser.add_argument('-d', "--description", action='store_true')
    parser.add_argument('-e', "--entity", action='store_true')
    parser.add_argument('-l', "--location", action='store_true')
    parser.add_argument('-n', "--numeric", action='store_true')
    parser.add_argument('-p', "--person", action='store_true')
    
    return parser

def get_categories_from_config(config):
    """
    Given parsed arguments, return a list of strings representing 
    the category types to use
    """
    categories = []
    if config.description:
        categories.append('description')
    if config.entity:
        categories.append('entity')
    if config.location:
        categories.append('location')
    if config.numeric:
        categories.append('numeric')
    if config.person:
        categories.append('person')

    if len(categories) == 0:
        raise ValueException('Indicate which answer types to use with command line arguments')

    return categories

def main():
    parser = get_parser()
    config = parser.parse_args()

    categories = get_categories_from_config(config)

    print('Using following answer types:', categories)

    queries = data.load_data(categories, config.train_dir)

    print(queries)

if __name__ == "__main__":
    main()