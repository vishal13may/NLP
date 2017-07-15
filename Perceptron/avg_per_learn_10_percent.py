import os
import glob
import sys
import math
from collections import defaultdict
from random import shuffle


total_ham_documents = 0
total_spam_documents = 0

vocabulary = defaultdict(lambda: [])
document_dictionary = {}
list_all_files = []

ham_directories = []
spam_directories = []

bias = 0
average_bias = 0
count = 1

document_label_dictionary = {}
CONSTANT_SPAM_LABEL = 1
CONSTANT_HAM_LABEL = -1

CONSTANT_HAM = "HAM"
CONSTANT_SPAM = "SPAM"
CONSTANT_FILE_WRITE_SEPARATOR = " "
CONSTANT_FILE_EXTENSION = ".txt"
CONSTANT_NB_MODEL = "per_model.txt"
CONSTANT_PERCENT = 10

NUMBER_OF_ITERATION = 30


def find_spam_ham_directories():
    # Function to find Spam and Ham directories in training data.
    for directory_name, directory_names, file_names in os.walk(training_data_path):
        if os.path.basename(directory_name).strip().upper() == CONSTANT_HAM:
            ham_directories.append(directory_name)
            global total_ham_documents
            total_ham_documents += len(glob.glob(os.path.join(directory_name, '*'+CONSTANT_FILE_EXTENSION)))
        elif os.path.basename(directory_name).strip().upper() == CONSTANT_SPAM:
            spam_directories.append(directory_name)
            global total_spam_documents
            total_spam_documents += len(glob.glob(os.path.join(directory_name, '*'+CONSTANT_FILE_EXTENSION)))
    return

def read_all_files(class_directories, class_file_count, label):
    # Function to generate Vocabulary
    file_counter = 0
    for directory in class_directories:
        if "1" in directory:
            continue
        if "3" in directory:
            continue
        for filename in glob.glob(os.path.join(directory, '*'+CONSTANT_FILE_EXTENSION)):
            try:
                if file_counter >= class_file_count:
                    return
                file_counter += 1
                list_all_files.append(filename)
                document_label_dictionary[filename] = label
                features = defaultdict(int)
                lines = [line for line in open(filename,"r",encoding="latin1")]
                for line in lines:
                    tokens = line.split()
                    for token in tokens:
                        vocabulary[token] = [0, 0]
                        features[token] += 1
                document_dictionary[filename] = features
            except IOError:
                continue
            except:
                continue
    return


def generate_model():
    # Function to generate Model
    global bias
    global count
    global average_bias

    for itr in range(NUMBER_OF_ITERATION):
        shuffle(list_all_files)
        for filename in list_all_files:
            try:
                y = document_label_dictionary[filename]
                features = document_dictionary[filename]
                frequency = 0
                for key in features:
                    values = vocabulary[key]
                    frequency += values[0]*features[key]

                alpha = frequency + bias
                if alpha*y <= 0:
                    for key in features:
                        values = vocabulary[key]
                        values[0] += (y*features[key])
                        values[1] += (y*features[key]*count)
                        vocabulary[key] = values
                    bias += y
                    average_bias += (y*count)
                count += 1
            except:
                continue

    average_bias = bias - (average_bias/count)
    for key in vocabulary:
        values = vocabulary[key]
        values[1] = values[0] - (values[1]/count)
        vocabulary[key] = values

    return

def write_training_model():
    #Write training model.
    try:
        with open(CONSTANT_NB_MODEL, 'w', encoding="latin1") as file_handler:
            file_handler.write(str(average_bias)+'\n')
            for key in vocabulary:
                values = vocabulary[key]
                file_handler.write(key+CONSTANT_FILE_WRITE_SEPARATOR+str(values[1])+'\n')
        file_handler.close()
    except IOError:
        pass
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please enter the training data directory.")
        sys.exit(0)

    training_data_path = sys.argv[1]
    training_data_path = os.path.abspath(training_data_path)

    find_spam_ham_directories()

    if len(spam_directories) == 0:
        print("Given path does not contain spam directories.")
        sys.exit(0)

    if len(ham_directories) == 0:
        print("Given path does not contain ham directories.")
        sys.exit(0)

    if total_spam_documents == 0:
        print("Given path does not contain spam files.")
        sys.exit(0)

    if total_spam_documents == 0:
        print("Given path does not contain ham files.")
        sys.exit(0)

    total_spam_documents = math.ceil((total_spam_documents * CONSTANT_PERCENT)/100)
    total_ham_documents = math.ceil((total_ham_documents * CONSTANT_PERCENT)/100)

    read_all_files(spam_directories, total_spam_documents, CONSTANT_SPAM_LABEL)
    read_all_files(ham_directories, total_ham_documents, CONSTANT_HAM_LABEL)

    generate_model()
    write_training_model()



