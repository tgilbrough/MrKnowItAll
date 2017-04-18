import json
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='./train_v1.1.json')
    parser.add_argument('--train_dir', default='./train/')
    parser.add_argument('--dev_path', default='./dev_v1.1.json')
    parser.add_argument('--dev_dir', default='./dev/')
    parser.add_argument('--test_path', default='./test_public_v1.1.json')
    parser.add_argument('--test_dir', default='./test/')
    
    return parser

def split(path, dest_dir):
    question_files = {}

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(path, encoding='utf-8') as data_file:    
        for line in data_file:
            question = json.loads(line)
            answer_type = question['query_type']

            # Initialization
            if answer_type not in question_files:
                question_files[answer_type] = open(dest_dir + answer_type + '.json', 'w', encoding='utf-8')

            # Print question out to appropriate file
            print(line, end='', file=question_files[answer_type])

        # Close all open files
        for answer_type in question_files.keys():
            question_files[answer_type].close()

def main():
    parser = get_parser()
    config = parser.parse_args()

    split(config.train_path, config.train_dir)
    split(config.dev_path, config.dev_dir)
    split(config.test_path, config.test_dir)

if __name__ == "__main__":
    main()

