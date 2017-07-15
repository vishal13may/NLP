import sys
import pycrfsuite
import hw3_corpus_tool
import csv
import glob
import os
import ntpath
from collections import defaultdict

FIRST_UTTERANCE = "F_U"
SPEAKER_CHANGE = "S_C"
tagged_data = defaultdict(lambda: [])
xList = []
yList = []

# For Calulations
correct = 0
total = 0


def write_to_file(output_file):
    with open(output_file, "w", encoding='latin1') as write_file_pointer:
        for file_name in tagged_data :
            write_file_pointer.write("Filename="+"\""+file_name+"\""+"\n")
            for label in tagged_data[file_name]:
                write_file_pointer.write(label+"\n")
            write_file_pointer.write("\n")
    return

def get_features(file_data):
    speaker = ""
    first_utterance = True
    features = []
    label = []
    for row in file_data:
        utterance = []
        label.append(row.act_tag)
        if first_utterance:
            utterance.append(FIRST_UTTERANCE)
            first_utterance = False
            speaker = row.speaker
        if speaker != row.speaker :
            utterance.append(SPEAKER_CHANGE)
        speaker = row.speaker
        if row.pos:
            utterance.extend(["TOKEN_" + tuple.token for tuple in row.pos])
            utterance.extend(["POS_" + tuple.pos for tuple in row.pos])
        features.append(utterance)
    return features,label

def train_model():
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.append(xList, yList)
    trainer.set_params({
        'c1': 1.0,
        'c2': 0.1,
        'max_iterations': 130,
        'feature.possible_transitions': True
    })
    trainer.train('baseline_model.crfsuite')
    return

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Please enter all required parameters.")
        sys.exit(0)

    INPUT_DIR = sys.argv[1]
    INPUT_DIR = os.path.abspath(INPUT_DIR)

    TEST_DIR = sys.argv[2]
    TEST_DIR = os.path.abspath(TEST_DIR)

    OUTPUT_FILE = sys.argv[3]

    all_files_data = hw3_corpus_tool.get_data(INPUT_DIR)

    # Read train data
    for file_data in all_files_data:
        features,act_tags = get_features(file_data)
        xList.extend(features)
        yList.extend(act_tags)

    train_model()

    #Read test data and Tag it
    tagger = pycrfsuite.Tagger()
    tagger.open('baseline_model.crfsuite')

    test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))
    for file_name in test_files:
        file_data = hw3_corpus_tool.get_utterances_from_filename(file_name)
        features,actual_labels = get_features(file_data)
        predicted_labels = tagger.tag(features)
        tagged_data[os.path.basename(file_name)] = predicted_labels
        #i = 0
        #for predicted_label in predicted_labels:
        #    if predicted_label == actual_labels[i]:
        #        correct += 1
        #    total += 1
        #    i += 1
    #print(correct,total)

    write_to_file(OUTPUT_FILE)





