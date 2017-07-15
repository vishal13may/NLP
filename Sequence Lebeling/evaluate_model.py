import sys
import re
import hw3_corpus_tool
import glob
import os
from collections import defaultdict

actual_labels = defaultdict(lambda: [])
predicted_labels = defaultdict(lambda: [])

def get_actual_labels(test_files):
    for file_name in test_files:
        file_data = hw3_corpus_tool.get_utterances_from_filename(file_name)
        label = []
        for row in file_data:
            label.append(row.act_tag)
        actual_labels[os.path.basename(file_name)] = label


    return

def read_predicted_labels(output_file):
    with open(output_file, "r", encoding="latin1") as read_handler:
                filename = ""
                for line in read_handler:
                    if "Filename" in line:
                        filename = re.findall(r'"([^"]*)"', line)[0]
                    elif line.strip() != '':
                       predicted_labels[filename].append(line.strip())
    return

def compare_labels():
    correct = 0
    total = 0
    for file_name in actual_labels :
        actual_list = actual_labels[file_name]
        predicted_list = predicted_labels[file_name]
        for i in range(0,len(actual_list)):
            if actual_list[i] == predicted_list[i]:
                correct += 1
            total += 1
    return correct,total

def calculate_accuracy(correct,total):
    accuracy = (correct / total) * 100
    return accuracy

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please enter all required parameters.")
        sys.exit(0)

    TEST_DIR = sys.argv[1]
    TEST_DIR = os.path.abspath(TEST_DIR)

    OUTPUT_FILE = sys.argv[2]
    OUTPUT_FILE = os.path.abspath(OUTPUT_FILE)

    test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))
    get_actual_labels(test_files)

    read_predicted_labels(OUTPUT_FILE)

    correct, total = compare_labels()

    accuracy = calculate_accuracy(correct,total)

    print(correct,total)

    print(accuracy)

