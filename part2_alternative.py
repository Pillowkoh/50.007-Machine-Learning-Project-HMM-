import pprint
import pandas as pd
import numpy as np
from part1 import emission_MLE

def update_transition(prev_tag, tag, transition_count):
    # Add prev_tag into transition_count dictionary
    if prev_tag not in transition_count:
        transition_count[prev_tag] = {}

    # Add next transition state into prev_tag dictionary
    if tag not in transition_count[prev_tag]:
        transition_count[prev_tag][tag] = 1
    else:
        transition_count[prev_tag][tag] += 1
    
    # Update prev_tag to tag
    prev_tag = tag
    return transition_count, prev_tag


def transition_MLE(filepath):
    file = open(filepath, 'r', encoding='UTF-8')
    separator = ' '

    states = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']

    tag_count = {
        'O': 0,
        'B-positive': 0, 
        'B-neutral': 0, 
        'B-negative': 0, 
        'I-positive': 0, 
        'I-neutral': 0, 
        'I-negative': 0,
        'START': 0,
        'STOP': 0
    }

    transition_count = {}

    prev_tag = None

    for line in file:

        if line != '\n':
            line = line.strip()
            token, tag = line.rsplit(sep=separator, maxsplit=1)

        # Handles the start of file
        if prev_tag == None and line != '\n':

            tag_count['START'] += 1
            prev_tag = 'START'
            tag_count[tag] += 1

            transition_count, prev_tag = update_transition(prev_tag, tag, transition_count)

        # Handles middle of sentence
        elif prev_tag in states and line != '\n':
            tag_count[tag] += 1
            transition_count, prev_tag = update_transition(prev_tag, tag, transition_count)           

        # Handle End of Sentence
        elif prev_tag in states and line == '\n':
            tag_count['STOP'] += 1
            tag = 'STOP'
            transition_count, prev_tag = update_transition(prev_tag, tag, transition_count)
            prev_tag = None

    transition_probability = transition_count

    # calculate the emission probability by dividing occurrence by total count
    for key, value in transition_probability.items():
        for key2, value2 in value.items():
            value[key2] /= tag_count[key]     

    # pprint.pprint(transition_probability)

    # Create Pandas Dataframe for visualisation purposes
    t_prob = pd.DataFrame.from_dict(transition_probability).T.fillna(0)
    # print(t_prob)

    return transition_probability


class Viterbi:

    def __init__(self, transition, emission):
        self.transition = pd.DataFrame.from_dict(transition).T.fillna(0)
        self.emission = pd.DataFrame.from_dict(emission).T.fillna(0)

    def calculate_score(self, current_index, current_state, score_array, observed_data=None):
        score_list = []
        states = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']
        for state in states:
            if current_index == 0:
                try:
                    score = 1 * self.transition.loc['START'][current_state] * self.emission.loc[current_state][observed_data]
                except:
                    score = 1 * self.transition.loc['START'][current_state] * self.emission.loc[current_state]['#UNK#']
                finally:
                    score_list.append(score)
            elif current_state == 'STOP':
                score = score_array.loc[current_index-1][state][0] * self.transition.loc[state][current_state]
                score_list.append(score) 
            else:
                try:
                    score = score_array.loc[current_index-1][state][0] * self.transition.loc[state][current_state] * self.emission.loc[current_state][observed_data]
                except:
                    score = score_array.loc[current_index-1][state][0] * self.transition.loc[state][current_state] * self.emission.loc[current_state]['#UNK#']
                finally:
                    score_list.append(score)
                    
        score = max(score_list)

        if current_index == 0:
            parent = 'START'
        else:
            parent = states[score_list.index(score)]

        return (score, parent)

    def forward(self, sentence):
        # assert that sentence is a list of strings
        # max score of each node at level j
        # pi(j+1, u) = max_v {pi(j, v) * b_u(x_j+1) * a_v,u}
        # number of states = 7

        predicted_path = []

        index = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']
        score_array = pd.DataFrame(np.zeros((7, len(sentence))), index=index, dtype=object).T

        for i in range(len(sentence)):
            for state in index:
                score_tuple = self.calculate_score(i, state, score_array, sentence[i])
                score_array.loc[i][state] = score_tuple

        print(f'Score array:\n {score_array}\n')

        stop_score = self.calculate_score(len(sentence), 'STOP', score_array)

        predicted_path.insert(0, stop_score[1])
        for i in range(len(sentence)):
            j = len(sentence) - i - 1
            predicted_path.insert(0, score_array.loc[j][predicted_path[0]][1])

        return predicted_path[1:]

    def predict(self, test_set):
        file = open(test_set, 'r', encoding='UTF-8')
        output = open(test_set + '/../dev.p2.out', 'w', encoding='UTF-8')

        sentences = []
        sentence = []
        for line in file:
            if line != '\n':
                line = line.strip()
                sentence.append(line)
            else:
                if len(sentence) != 0:
                    sentences.append(sentence)
                    sentence = []
        
        for i in sentences:
            prediction = self.forward(i)
            predictions = [a + ' ' + b + '\n' for a,b in zip(i, prediction)]
            for j in predictions:
                output.write(j)
            output.write('\n')

        output.close()





# index = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']
# score_array = pd.DataFrame(np.zeros((7, 10)), index=index).T

# score_array.loc[0]['O'] = 1

# print(score_array)


# filepath = './ES/ES/train'
# t_prob = transition_MLE(filepath=filepath)
# e_prob = emission_MLE(filepath=filepath)

# viterbi = Viterbi(t_prob, e_prob)

# # sentence =['La',
# # 'comida',
# # 'estuvo',
# # 'muy',
# # 'sabrosa',
# # '.']

# # prediction = viterbi.forward(sentence=sentence)
# # print(prediction)

# test_set = './ES/ES/dev.in'
# viterbi.predict(test_set=test_set)

