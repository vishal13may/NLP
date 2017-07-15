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

# Caclculations
correctly_classified_spam = 0
correctly_classified_ham = 0
classified_as_spam = 0
classified_as_ham = 0
actual_spam = 0
actual_ham = 0

def calculations():
    #funcntion to calculate Precision, Recall , F1 and Weighted Average
    print("Correctly Classified Spam :"+str(correctly_classified_spam))
    print("Correctly Classified Ham :"+str(correctly_classified_ham))
    print("Classified as Spam :"+ str(classified_as_spam))
    print("Classified as Ham :"+str(classified_as_ham))
    print("Actual Spam :"+str(actual_spam))
    print("Actual Ham :"+str(actual_ham))

    accuracy = (correctly_classified_spam + correctly_classified_ham) / (actual_ham + actual_spam)
    print("Accuracy :"+str(accuracy))

    precision_spam = correctly_classified_spam / classified_as_spam
    print("Precision(spam):"+str(precision_spam))

    recall_spam = correctly_classified_spam / actual_spam
    print("Recall(spam):"+str(recall_spam))

    f1_spam = (2 * precision_spam * recall_spam ) / (precision_spam + recall_spam)
    print("F1(spam) :"+str(f1_spam))

    precision_ham = correctly_classified_ham / classified_as_ham
    print("Precision(ham):"+str(precision_ham))

    recall_ham = correctly_classified_ham / actual_ham
    print("Recall(ham):"+str(recall_ham))

    f1_ham = (2 * precision_ham * recall_ham ) / (precision_ham + recall_ham)
    print("F1(ham) :"+str(f1_ham))

    weighed_average = (f1_spam + f1_ham) / 2
    print("Weighted Average :"+str(weighed_average))

    return


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

    global correctly_classified_spam
    global correctly_classified_ham
    global classified_as_spam
    global classified_as_ham
    global actual_spam
    global actual_ham

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
                        classified_as_spam += 1
                        if "ham" in filename:
                            actual_ham += 1
                        else:
                            correctly_classified_spam += 1
                            actual_spam += 1
                    else :
                        write_file_pointer.write("HAM"+" "+filename+"\n")
                        classified_as_ham += 1
                        if "spam" in filename:
                            actual_spam += 1
                        else:
                            correctly_classified_ham += 1
                            actual_ham += 1
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
    calculations()
