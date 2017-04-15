import json
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default='./train_v1.1.json')
    parser.add_argument("--dest-dir", default='./train/')
    
    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()

    question_files = {}

    if not os.path.exists(config.dest_dir):
        os.makedirs(config.dest_dir)

    with open(config.data_file, encoding='utf-8') as data_file:    
        for line in data_file:
            question = json.loads(line)
            answer_type = question['query_type']

            # Initialization
            if answer_type not in question_files:
                question_files[answer_type] = open(config.dest_dir + answer_type + '.json', 'w', encoding='utf-8')

            # Print question out to appropriate file
            print(line, end='', file=question_files[answer_type])

        # Close all open files
        for answer_type in question_files.keys():
            question_files[answer_type].close()


if __name__ == "__main__":
    main()

