import os
import glob
import sys


total_distinct_words = 0

total_words_ham = 0
ham_directories = []
total_ham_documents = 0

total_words_spam = 0
spam_directories = []
total_spam_documents = 0

CONSTANT_HAM = "HAM"
CONSTANT_SPAM = "SPAM"
CONSTANT_FILE_WRITE_SEPARATOR = " "
CONSTANT_NB_MODEL = "nbmodel.txt"
CONSTANT_FILE_EXTENSION = ".txt"

SPECIAL_CHARACTER = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']
STOP_WORDS = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

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

def generate_vocabulary(class_type,class_directories):
    # Function to generate Vocabulary
    for directory in class_directories:
        for filename in glob.glob(os.path.join(directory, '*'+CONSTANT_FILE_EXTENSION)):
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
                    if token in STOP_WORDS:
                        continue
                    if token in SPECIAL_CHARACTER:
                        continue
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

    generate_vocabulary(CONSTANT_SPAM, spam_directories)
    if total_words_spam == 0:
        print("No Spam data available");
        sys.exit(0)

    generate_vocabulary(CONSTANT_HAM, ham_directories)
    if total_words_ham == 0:
        print("No Ham data available");
        sys.exit(0)

    total_documents = total_spam_documents + total_ham_documents

    sorted_spam = sorted(vocabulary, key=lambda k: vocabulary[k][0], reverse=True)
    sorted_ham = sorted(vocabulary, key=lambda k: vocabulary[k][1], reverse=True)
    sorted_spam_list = sorted_spam[:100]
    sorted_ham_list = sorted_spam[:100]

    commonwords_list = list(set(sorted_spam_list).intersection(sorted_ham_list))

    for word in commonwords_list:
        del vocabulary[word]

    vocabulary_size = len(vocabulary)
    write_training_model()




