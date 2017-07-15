import sys
import pycrfsuite
import hw3_corpus_tool
import glob
import os
from collections import defaultdict



FIRST_UTTERANCE = "F_U"
SPEAKER_CHANGE = "S_C"
PREV_TOKEN = "TOKENPREV"
PREV_POS = "POSPREV"
LAST_UTTERENCE = "L_U"
tagged_data = defaultdict(lambda: [])


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Please enter all required parameters.")
        sys.exit(0)

    INPUT_DIR = sys.argv[1]
    INPUT_DIR = os.path.abspath(INPUT_DIR)

    TEST = sys.argv[2]
    TEST = os.path.abspath(TEST)

    CONSTANT_FILE_NB_OUTPUT = sys.argv[3]

    all_files_data = hw3_corpus_tool.get_data(INPUT_DIR)

    xList = []
    yList = []

    for file_data in all_files_data:
        speaker = ""
        last_token = ""
        last_pos = ""
        first_utterance = True
        for count, row in enumerate(file_data):
            utterance = []
            yList.append(row.act_tag)
            if first_utterance:
                utterance.append(FIRST_UTTERANCE)
                first_utterance = False
                speaker = row.speaker
            if speaker != row.speaker :
                utterance.append(SPEAKER_CHANGE)
            speaker = row.speaker
            if row.pos:
                length = len(row.pos)
                prev_token = row.pos[0].token
                prev_pos = row.pos[0].pos
                utterance.append("TOKEN_"+row.pos[0].token)
                utterance.append("POS_"+row.pos[0].pos)
                for i,tuple in enumerate(row.pos[1: -1]):
                    utterance.append(PREV_TOKEN+"_"+prev_token+"|"+tuple.token)
                    utterance.append(PREV_POS+"_"+prev_pos+"|"+tuple.pos)
                    utterance.append("TOKEN_"+tuple.token)
                    utterance.append("POS_"+tuple.pos)
                    prev_token = tuple.token
                    prev_pos = tuple.pos
                    try:
                        utterance.append("TOK_"+row.pos[i-1].token+"|"+row.pos[i].token+"|"+row.pos[i+1].token)
                        utterance.append("POST_"+row.pos[i-1].pos+"|"+row.pos[i].pos+"|"+row.pos[i+1].pos)
                    except:
                        pass

                utterance.append("TOKENLEN_"+str(length))
                utterance.append("POSLEN_"+str(length))

            xList.append(utterance)
        xList[len(xList) - 1].append(LAST_UTTERENCE)
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.append(xList, yList)
    trainer.set_params({
        'max_iterations': 100,
        'c1': 1.0,
        'c2': 1e-3,
		'feature.possible_transitions': True
    })

    trainer.train('advanced_model.crfsuite')

    ######  Prediction ####

    tagger = pycrfsuite.Tagger()
    tagger.open('advanced_model.crfsuite')


    correct = 0
    total = 0

    test_filenames = glob.glob(os.path.join(TEST, "*.csv"))
    for file_name in test_filenames:
        file_data = hw3_corpus_tool.get_utterances_from_filename(file_name)
        speaker = ""
        first_utterance = True
        input_list = []
        actual_label = []
        for row in file_data:
            utterance = []
            actual_label.append(row.act_tag)
            if first_utterance:
                utterance.append(FIRST_UTTERANCE)
                first_utterance = False
                speaker = row.speaker
            if speaker != row.speaker :
                utterance.append(SPEAKER_CHANGE)
            speaker = row.speaker
            if row.pos:
                length = len(row.pos)
                prev_token = row.pos[0].token
                prev_pos = row.pos[0].pos
                utterance.append("TOKEN_"+row.pos[0].token)
                utterance.append("POS_"+row.pos[0].pos)
                for i,tuple in enumerate(row.pos[1: -1]):
                    utterance.append(PREV_TOKEN+"_"+prev_token+"|"+tuple.token)
                    utterance.append(PREV_POS+"_"+prev_pos+"|"+tuple.pos)
                    utterance.append("TOKEN_"+tuple.token)
                    utterance.append("POS_"+tuple.pos)
                    prev_token = tuple.token
                    prev_pos = tuple.pos
                    try:
                        utterance.append("TOK_"+row.pos[i-1].token+"|"+row.pos[i].token+"|"+row.pos[i+1].token)
                        utterance.append("POST_"+row.pos[i-1].pos+"|"+row.pos[i].pos+"|"+row.pos[i+1].pos)
                    except:
                        pass

                utterance.append("TOKENLEN_"+str(length))
                utterance.append("POSLEN_"+str(length))

            input_list.append(utterance)
        input_list[len(input_list) - 1].append(LAST_UTTERENCE)
        predicted_labels = tagger.tag(input_list)
        tagged_data[os.path.basename(file_name)] = predicted_labels
 
    # Write Data

    with open(CONSTANT_FILE_NB_OUTPUT, "w", encoding='latin1') as write_file_pointer:
        for file_name in tagged_data :
            write_file_pointer.write("Filename="+"\""+file_name+"\""+"\n")
            for label in tagged_data[file_name]:
                write_file_pointer.write(label+"\n")
            write_file_pointer.write("\n")
        

