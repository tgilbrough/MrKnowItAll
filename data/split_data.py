import json
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='./train_v1.1.json')
    parser.add_argument("--dest_dir", default='./train/')
    parser.add_argument("--max_q", type=int, default=1000)
    
    return parser

def main():
    parser = get_parser()
    config = parser.parse_args()

    question_count = {}
    question_files = {}

    with open(config.data_file, encoding='utf-8') as data_file:    
        for line in data_file:
            question = json.loads(line)
            answer_type = question['query_type']

            # Initialization
            if answer_type not in question_files:
                question_count[answer_type] = 0
                question_files[answer_type] = open(config.dest_dir + answer_type + '_0.json', 'w', encoding='utf-8')

            # Print question out to appropriate file
            print(line, end='', file=question_files[answer_type])

            question_count[answer_type] += 1

            # If hit the max questions per file, close it and open a new one 
            if question_count[answer_type] % config.max_q == 0:
                question_files[answer_type].close()
                question_files[answer_type] = open(config.dest_dir 
                    + answer_type + '_' + str(question_count[answer_type] // config.max_q) + '.json', 'w',  encoding='utf-8')

        # Close all open files
        for answer_type in question_files.keys():
            question_files[answer_type].close()


if __name__ == "__main__":
    main()

