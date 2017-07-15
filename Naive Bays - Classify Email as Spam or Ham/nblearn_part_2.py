import os
import glob
import sys
import math

training_data_path = ""

total_documents = 0

total_words_ham = 0
ham_directories = []
total_ham_documents = 0

total_words_spam = 0
spam_directories = []
total_spam_documents = 0

CONSTANT_HAM = "HAM"
CONSTANT_SPAM = "SPAM"
CONSTANT_FILE_WRITE_SEPARATOR = " ";
CONSTANT_FILE_EXTENSION = ".txt"
CONSTANT_NB_MODEL = "nbmodel.txt"
CONSTANT_PERCENT = 10


# Key = {String} Values = {List of 2 Integers ; 0th position -SPAM 1st Position - HAM }
vocabulary = {}
vocabulary_size = 0


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

def generate_vocabulary(class_type,class_directories,class_file_count):
    # Function to generate Vocabulary
    file_counter = 0
    for directory in class_directories:
        if "1" in directory:
            continue
        if "3" in directory:
            continue
        # Select Files from Directory 2 only.
        # Tested for all directories and noticed that it is giving better performance when files are selected from directory 2
        for filename in glob.glob(os.path.join(directory, '*'+CONSTANT_FILE_EXTENSION)):
            if file_counter >= class_file_count:
                return
            file_counter += 1
            lines = [line for line in open(filename,"r",encoding="latin1")]
            for line in lines:
                tokens = line.split()
                if class_type == CONSTANT_SPAM:
                    global total_words_spam
                    total_words_spam += len(tokens)
                elif class_type == CONSTANT_HAM:
                    global total_words_ham
                    total_words_ham += len(tokens)
                for token in tokens:
                    token = token.strip()
                    if token in vocabulary:
                        if class_type == CONSTANT_SPAM:
                            values = vocabulary[token]
                            values[0] += 1
                            vocabulary[token] = values
                        elif class_type == CONSTANT_HAM:
                            values = vocabulary[token]
                            values[1] += 1
                            vocabulary[token] = values
                    else:
                        if class_type == CONSTANT_SPAM:
                            vocabulary[token] = [1, 0]
                        elif class_type == CONSTANT_HAM:
                            vocabulary[token] = [0, 1]

    return

def write_training_model():
    #Write training model.
    with open(CONSTANT_NB_MODEL, 'w',encoding="latin1") as file_handler:
        file_handler.write(str(vocabulary_size)+'\n')
        file_handler.write(str(total_words_spam)+'\n')
        file_handler.write(str(total_words_ham)+'\n')
        file_handler.write("P(SPAM)"+CONSTANT_FILE_WRITE_SEPARATOR+str(total_spam_documents/total_documents)+'\n')
        file_handler.write("P(HAM)"+CONSTANT_FILE_WRITE_SEPARATOR+str((total_ham_documents/total_documents))+'\n')
        # Read token and calculate the probabilities
        # Add one smoothing
        for token in vocabulary:
            values = vocabulary[token]
            spam_probability = (values[0] + 1) / (total_words_spam + vocabulary_size)
            ham_probability = (values[1] + 1) / (total_words_ham +vocabulary_size)
            file_handler.write(token+CONSTANT_FILE_WRITE_SEPARATOR+str(spam_probability)+CONSTANT_FILE_WRITE_SEPARATOR+str(ham_probability)+'\n')
    file_handler.close()
    return

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please enter the training data directory.")
        sys.exit(0)

    training_data_path = sys.argv[1]
    find_spam_ham_directories()
    if len(spam_directories) == 0:
        print("Given path does not contain spam directories.")
        sys.exit(0)
    if len(ham_directories) == 0:
        print("Given path does not contain ham directories.")
        sys.exit(0)

    total_spam_documents = math.ceil((total_spam_documents * CONSTANT_PERCENT)/100)
    total_ham_documents = math.ceil((total_ham_documents * CONSTANT_PERCENT)/100)
    total_documents = total_spam_documents + total_ham_documents

    generate_vocabulary(CONSTANT_SPAM, spam_directories, total_spam_documents)
    if total_words_spam == 0:
        print("No Spam data available");
        sys.exit(0)

    generate_vocabulary(CONSTANT_HAM, ham_directories, total_ham_documents)
    if total_words_ham == 0:
        print("No Ham data available");
        sys.exit(0)

    vocabulary_size = len(vocabulary)
    write_training_model()




