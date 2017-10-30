# Competition 1
import pandas as pd
import numpy as np
import jieba

jieba.set_dictionary('big5_dict.txt')


def jieba_lines(lines):
    cut_lines = []

    for line in lines:
        cut_line = jieba.lcut(line)
        cut_lines.append(cut_line)

    return cut_lines


def add_word_dict(w):
    if not w in word_dict:
        word_dict[w] = 1
    else:
        word_dict[w] += 1


def generate_training_data():
    Xs, Ys = [], []

    for i in range(NUM_TRAIN):
        pos_or_neg = random.randint(0, 1)

        if pos_or_neg == 1:
            program_id = random.randint(0, NUM_PROGRAM - 1)
            episode_id = random.randint(0, len(cut_programs[program_id]) - 1)
            line_id = random.randint(0, len(cut_programs[program_id][episode_id]) - 2)

            Xs.append([cut_programs[program_id][episode_id][line_id],
                       cut_programs[program_id][episode_id][line_id + 1]])
            Ys.append(1)

        else:
            first_program_id = random.randint(0, NUM_PROGRAM - 1)
            first_episode_id = random.randint(0, len(cut_programs[first_program_id]) - 1)
            first_line_id = random.randint(0, len(cut_programs[first_program_id][first_episode_id]) - 1)

            second_program_id = random.randint(0, NUM_PROGRAM - 1)
            second_episode_id = random.randint(0, len(cut_programs[second_program_id]) - 1)
            second_line_id = random.randint(0, len(cut_programs[second_program_id][second_episode_id]) - 1)

            Xs.append([cut_programs[first_program_id][first_episode_id][first_line_id],
                       cut_programs[second_program_id][second_episode_id][second_line_id]])
            Ys.append(0)

    return Xs, Ys


if __name__ == '__main__':

    ###### Feature Engineering ######

    # Read in the programs.
    NUM_PROGRAM = 8
    programs = []
    for i in range(1, NUM_PROGRAM + 1):
        program = pd.read_csv('Program0%d.csv' % (i))

        print('Program %d' % (i))
        print('Episodes: %d' % (len(program)))
        print(program.columns)
        print(program.loc[:1]['Content'])
        print()

        programs.append(program)

    # Read in the questions.
    questions = pd.read_csv('Question.csv')

    print('Question')
    print('Episodes: %d' % (len(questions)))
    print(questions.columns)
    print(questions.loc[:2]['Question'])
    print()

    NUM_OPTION = 6
    for i in range(NUM_OPTION):
        print(questions.loc[:2]['Option%d' % (i)])
        print()

    # lines = programs[0].loc[1]['Content'].split('\n')
    # for line in lines:
    #     print(jieba.lcut(line))

    #
    ###### Preprocessing: Cut Words ######
    # Since chinese characters are continuous one by one, we have to cut them into meaningful words first.
    # We use jieba with traditional chinese dictionary to cut our text.

    # We cut not only Program.csv but also Question.csv, and save as list.
    cut_programs = np.load('cut_Programs.npy')
    cut_questions = np.load('cut_Questions.npy')

    print(len(cut_programs))
    print(len(cut_programs[0]))
    print(len(cut_programs[0][0]))
    print(cut_programs[0][0][:3])

    print(len(cut_questions))
    print(len(cut_questions[0]))
    print(cut_questions[0][0])

    for i in range(1, 7):
        print(cut_questions[0][i])

    # Preprocessing: Word Dictionary & Out-of-Vocabulary
    # There are many words after cutting, but not all of them is useful.
    # The word too common or too rare can not give us information but may noise.
    # We count the the number of occurrence for each word and remove useless ones.
    voc_dict = np.load('voc_dict.npy')
    # Now, voc_dict becomes better word dictionary, then we should replace those removed words
    # aka out-of-vocabulary words into an unknown token in the following use.


    #
    ###### Preprocessing: Generating Training Data ######
    # Though the format of question is to select one from six, our traing data only have continuous lines.
    # Naively, i want to change the whole problem into a binary classification which means given two lines,
    # my model want to judge these two are context or not.

    import random

    NUM_TRAIN = 10000
    TRAIN_VALID_RATE = 0.7
    Xs, Ys = generate_training_data()

    x_train, y_train = Xs[:int(NUM_TRAIN * TRAIN_VALID_RATE)], Ys[:int(NUM_TRAIN * TRAIN_VALID_RATE)]
    x_valid, y_valid = Xs[int(NUM_TRAIN * TRAIN_VALID_RATE):], Ys[int(NUM_TRAIN * TRAIN_VALID_RATE):]


    example_doc = []

    for line in cut_programs[0][0]:
        example_line = ''
        for w in line:
            if w in voc_dict:
                example_line += w + ' '

        example_doc.append(example_line)

    print(example_doc[:10])

