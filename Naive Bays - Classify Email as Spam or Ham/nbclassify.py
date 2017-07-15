import os
import glob
import math
import sys

dev_data_path = ""

spam_probability = 0
ham_probability = 0

total_words_spam = 0
total_words_ham = 0
vocabulary_size = 0

CONSTANT_FILE_WRITE_SEPARATOR = " ";
CONSTANT_FILE_EXTENSION = ".txt"

CONSTANT_FILE_NB_MODEL = "nbmodel.txt"

CONSTANT_FILE_NB_OUTPUT = "nboutput.txt"



vocabulary = {}
vocabulary_size = 0


def generate_vocabulary(nb_model):
    # Function to generate Vocabulary
        with open(nb_model,'r', encoding="latin1") as file_pointer:
            global vocabulary_size
            vocabulary_size = float(file_pointer.readline())
            global total_words_spam
            total_words_spam = float(file_pointer.readline())
            global total_words_ham
            total_words_ham = float(file_pointer.readline())
            global spam_probability
            spam_probability = float(file_pointer.readline().split()[1])
            global ham_probability
            ham_probability = float(file_pointer.readline().split()[1])
            for line in file_pointer:
                tokens = line.split()
                vocabulary[tokens[0]] = [float(tokens[1]), float(tokens[2])]
        return

def classify():
    #Function to classify the documents
    with open(CONSTANT_FILE_NB_OUTPUT, "w", encoding='latin1') as write_file_pointer:
        for directory_name, directory_names, file_names in os.walk(dev_data_path):
            for filename in glob.glob(os.path.join(directory_name, '*'+CONSTANT_FILE_EXTENSION)):
                message_given_spam_prob = 0
                message_given_ham_prob = 0
                try:
                    with open(filename,'r',encoding="latin1") as file_pointer:
                        for line in file_pointer:
                            tokens = line.split()
                            for token in tokens:
                                token = token.strip()
                                if token in vocabulary:
                                    values = vocabulary[token]
                                    if values[0] >= 0:
                                        message_given_spam_prob += math.log(values[0])
                                    if values[1] >= 0:
                                        message_given_ham_prob += math.log(values[1])
                        if spam_probability >= 0:
                            prob_spam = math.log(spam_probability) + message_given_spam_prob
                        else:
                            prob_spam =  message_given_spam_prob
                        if ham_probability>= 0:
                            prob_ham = math.log(ham_probability) + message_given_ham_prob
                        else:
                            prob_ham = message_given_ham_prob

                    if prob_spam > prob_ham:
                        write_file_pointer.write("SPAM" + " "+filename+"\n")

                    elif prob_spam < prob_ham:
                        write_file_pointer.write("HAM" + " "+filename+"\n")
                    else:
                        write_file_pointer.write("SPAM"+" "+filename+"\n")
                except IOError:
                    continue
                except:
                    continue

    return

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please enter the training data directory.")
        sys.exit(0)

    dev_data_path = sys.argv[1]

    dev_data_path = os.path.abspath(dev_data_path)

    generate_vocabulary(CONSTANT_FILE_NB_MODEL)
    classify()



