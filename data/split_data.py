import json

source_file = './train_v1.1.json'
destination_dir = './train/'

max_questions_per_file = 1000

question_count = {}
question_files = {}

with open(source_file) as data_file:    
    for line in data_file:
        question = json.loads(line)
        answer_type = question['query_type']

        # Initialization
        if answer_type not in question_files:
            question_count[answer_type] = 0
            question_files[answer_type] = open(destination_dir + answer_type + '_0.json', 'w')

        # Print question out to appropriate file
        print(line, end='', file=question_files[answer_type])

        question_count[answer_type] += 1

        # If hit the max questions per file, close it and open a new one 
        if question_count[answer_type] % max_questions_per_file == 0:
            question_files[answer_type].close()
            question_files[answer_type] = open(destination_dir + answer_type + '_' + str(question_count[answer_type] // max_questions_per_file) + '.json', 'w')

    # Close all open files
    for answer_type in question_files.keys():
        question_files[answer_type].close()
