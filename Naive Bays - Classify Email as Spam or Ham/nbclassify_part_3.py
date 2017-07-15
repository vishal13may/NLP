import os
import glob
import math
import sys


spam_probability = 0
ham_probability = 0

total_words_spam = 0
total_words_ham = 0
vocabulary_size = 0

CONSTANT_FILE_WRITE_SEPARATOR = " ";
CONSTANT_FILE_EXTENSION = ".txt"

CONSTANT_FILE_NB_MODEL = "nbmodel.txt"

CONSTANT_FILE_NB_OUTPUT = "nboutput.txt"


# Key = {String} Values = {List of 2 Integers ; 0th position -SPAM 1st Position - HAM }
vocabulary = {}
vocabulary_size = 0

# Caclculations
correctly_classified_spam = 0
correctly_classified_ham = 0
classified_as_spam = 0
classified_as_ham = 0
actual_spam = 0
actual_ham = 0


def generate_vocabulary(nb_model):
    # Function to generate Vocabulary
        with open(nb_model,'r',encoding="latin1") as file_pointer:
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
    #Function to classify files.
    wrong_ham = 0;
    wrong_spam = 0;
    with open(CONSTANT_FILE_NB_OUTPUT, "w", encoding='latin1') as write_file_pointer:
        for directory_name, directory_names, file_names in os.walk(dev_data_path):
            for filename in glob.glob(os.path.join(directory_name, '*'+CONSTANT_FILE_EXTENSION)):
                message_given_spam_prob = 0
                message_given_ham_prob = 0
                with open(filename,'r',encoding="latin1") as file_pointer:
                    for line in file_pointer:
                        tokens = line.split()
                        for token in tokens:
                            token = token.strip()
                            if token in vocabulary:
                                values = vocabulary[token]
                                message_given_spam_prob += math.log(values[0])
                                message_given_ham_prob += math.log(values[1])

                prob_spam = math.log(spam_probability) + message_given_spam_prob
                prob_ham = math.log(ham_probability) + message_given_ham_prob

                if "ham" in filename:
                    global actual_ham
                    actual_ham += 1
                elif "spam" in filename:
                    global actual_spam
                    actual_spam += 1

                if prob_spam > prob_ham:
                    write_file_pointer.write("SPAM" + " "+filename+"\n")
                    global classified_as_spam
                    classified_as_spam += 1
                    if "ham" in filename:
                        wrong_spam +=1
                    else:
                        global correctly_classified_spam
                        correctly_classified_spam += 1
                elif prob_spam < prob_ham:
                    write_file_pointer.write("HAM" + " "+filename+"\n")
                    global classified_as_ham
                    classified_as_ham += 1
                    if "spam" in filename:
                        wrong_ham +=1
                    else:
                        global correctly_classified_ham
                        correctly_classified_ham += 1
                else:
                    write_file_pointer.write("SPAM"+" "+filename+"\n")

    print("Ham but detected Spam :" + str(wrong_spam))
    print("Spam but detected Ham :"+ str(wrong_ham))
    return

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

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please enter the training data directory.")
        sys.exit(0)

    dev_data_path = sys.argv[1]
    generate_vocabulary(CONSTANT_FILE_NB_MODEL)
    classify()
    calculations()






