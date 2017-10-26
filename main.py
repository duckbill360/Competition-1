# Competition 1
import pandas as pd
import jieba


def jieba_lines(lines):
    cut_lines = []

    for line in lines:
        cut_line = jieba.lcut(line)
        cut_lines.append(cut_line)

    return cut_lines




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
    #

    # Since chinese characters are continuous one by one, we have to cut them into meaningful words first.
    # We use jieba with traditional chinese dictionary to cut our text.

    # We cut not only Program.csv but also Question.csv, and save as list.
    cut_programs = []

    for program in programs:
        n = len(program)
        cut_program = []

        for i in range(n):
            lines = program.loc[i]['Content'].split('\n')
            cut_program.append(jieba_lines(lines))

        cut_programs.append(cut_program)

    print(len(cut_programs))
    print(len(cut_programs[0]))
    print(len(cut_programs[0][0]))
    print(cut_programs[0][0][:3])
