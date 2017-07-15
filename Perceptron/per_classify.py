import os
import glob
import sys
from collections import defaultdict

dev_data_path = ""

bias = 0

CONSTANT_FILE_WRITE_SEPARATOR = " "
CONSTANT_FILE_EXTENSION = ".txt"

CONSTANT_FILE_MODEL = "per_model.txt"
CONSTANT_FILE_OUTPUT = ""

vocabulary = {}

def generate_vocabulary(model):
    # Function to generate Vocabulary
        with open(model, 'r', encoding="latin1") as file_pointer:
            global bias
            bias = float(file_pointer.readline())
            for line in file_pointer:
                tokens = line.split()
                vocabulary[tokens[0]] = float(tokens[1])
        return

def classify():
    #Function to classify the documents

    with open(CONSTANT_FILE_OUTPUT, "w", encoding='latin1') as write_file_pointer:
        for directory_name, directory_names, file_names in os.walk(dev_data_path):
            for filename in glob.glob(os.path.join(directory_name, '*'+CONSTANT_FILE_EXTENSION)):
                features = defaultdict(int)
                try:
                    with open(filename,'r',encoding="latin1") as file_pointer:
                        for line in file_pointer:
                            tokens = line.split()
                            for token in tokens:
                                features[token] += 1

                    frequency = 0
                    for key in features:
                        if key in vocabulary:
                            frequency += (vocabulary[key]*features[key])

                    if frequency + bias > 0:
                        write_file_pointer.write("SPAM"+" "+filename+"\n")
                    else:
                        write_file_pointer.write("HAM"+" "+filename+"\n")
                except IOError:
                   continue
                except:
                    continue
    return

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please enter the training data directory.")
        sys.exit(0)

    dev_data_path = sys.argv[1]

    CONSTANT_FILE_OUTPUT = sys.argv[2]

    dev_data_path = os.path.abspath(dev_data_path)

    generate_vocabulary(CONSTANT_FILE_MODEL)
    classify()
