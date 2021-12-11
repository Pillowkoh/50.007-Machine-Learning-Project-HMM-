import pandas as pd
import numpy as np
import pprint
from part1 import emission_MLE
from part2_alternative import transition_MLE

class KBestViterbi:

    def __init__(self, transition, emission, k):
        self.transition = pd.DataFrame.from_dict(transition).T.fillna(0)
        self.emission = pd.DataFrame.from_dict(emission).T.fillna(0)
        self.k = k

    def calculate_score(self, current_index, current_state, score_array, observed_data=None):
        score_list = []
        states = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']
        #first layer (after START)
        if current_index == 0:
            try:
                score = 1 * self.transition.loc['START'][current_state] * self.emission.loc[current_state][observed_data]
            except:
                score = 1 * self.transition.loc['START'][current_state] * self.emission.loc[current_state]['#UNK#']
            finally:
                score_list.append(score)
        else:
            for state in states:
                if current_state == 'STOP':
                    for i in range(len(score_array.loc[current_index-1][state])):
                        score = score_array.loc[current_index-1][state][i] * self.transition.loc[state][current_state]
                        score_list.append(score) 
                else:
                    for i in range(len(score_array.loc[current_index-1][state])):
                        try:
                            score = score_array.loc[current_index-1][state][i] * self.transition.loc[state][current_state] * self.emission.loc[current_state][observed_data]
                        except:
                            score = score_array.loc[current_index-1][state][i] * self.transition.loc[state][current_state] * self.emission.loc[current_state]['#UNK#']
                        finally:
                            score_list.append(score)
                    
        if(current_index==0): #score list will only have 1 element
            return score_list

        score_list.sort(reverse=True)
        #print(score_list[:self.k])

        return score_list[:self.k]

    def forward(self, sentence):
        # assert that sentence is a list of strings
        # max score of each node at level j
        # pi(j+1, u) = max_v {pi(j, v) * b_u(x_j+1) * a_v,u}
        # number of states = 7

        index = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']
        score_array = pd.DataFrame(np.zeros((7, len(sentence))), index=index, dtype=object).T

        for i in range(len(sentence)):
            for state in index:
                score_tuple = self.calculate_score(i, state, score_array, sentence[i])
                score_array.loc[i][state] = score_tuple
                #print(score_tuple)

        #print(f'Score array:\n {score_array}\n')

        stop_score = self.calculate_score(len(sentence), 'STOP', score_array) #list with k elements, best prob first
        print(score_array)
        
        return score_array,stop_score

        # predicted_path.insert(0, stop_score[1])
        # for i in range(len(sentence)):
        #     j = len(sentence) - i - 1
        #     predicted_path.insert(0, score_array.loc[j][predicted_path[0]][1])

        # return predicted_path[1:]

    def backward(self,score_array,stop_score,sentence):
        predicted_paths = [] #list of lists, 0 is best path

        for sscore in stop_score: 
            predicted_paths.append(self.backwards_recursive(score_array,len(sentence)-1,'STOP',sscore,sentence))

        return predicted_paths
            

    def backwards_recursive(self,score_array,current_index,next_state,score_to_compare,sentence):
        #next state gives us score to compare
        states = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']

        if current_index == 0: #first word
            predicted_path = []
            for state in states:
                for path_prob in score_array.loc[current_index][state]:
                    try:
                        if(next_state=='STOP'):
                            if(score_to_compare == path_prob*self.transition.loc[state]['STOP']):
                                predicted_path.append(state)
                                return predicted_path
                        else:
                            if(score_to_compare == path_prob*self.transition.loc[state][next_state]*self.emission.loc[next_state][sentence[current_index+1]]):
                                predicted_path.append(state)
                                return predicted_path
                    except:
                        if(score_to_compare == path_prob*self.transition.loc[state][next_state]*self.emission.loc[next_state]['#UNK#']):
                            predicted_path.append(state)
                            return predicted_path

        elif current_index == len(sentence)-1: #last word
            for state in states:
                for path_prob in score_array.loc[current_index][state]:
                    if(score_to_compare == path_prob*self.transition.loc[state]['STOP']):
                        predicted_path = self.backwards_recursive(score_array,current_index-1,state,path_prob,sentence)
                        predicted_path.append(state)
                        return predicted_path

        else: #everyth in btwn
            for state in states:
                for path_prob in score_array.loc[current_index][state]:
                    try:
                        if(score_to_compare == path_prob*self.transition.loc[state][next_state]*self.emission.loc[next_state][sentence[current_index+1]]):
                            predicted_path = self.backwards_recursive(score_array,current_index-1,state,path_prob,sentence) 
                            predicted_path.append(state)
                            return predicted_path
                    except:
                        if(score_to_compare == path_prob*self.transition.loc[state][next_state]*self.emission.loc[next_state]['#UNK#']):
                            predicted_path = self.backwards_recursive(score_array,current_index-1,state,path_prob,sentence) 
                            predicted_path.append(state)
                            return predicted_path

    def predict(self, test_set):
        file = open(test_set, 'r', encoding='UTF-8')
        output = open(test_set + '/../dev.p3.out', 'w', encoding='UTF-8')

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
            score_array,stop_score = self.forward(i)
            prediction = self.backward(score_array,stop_score,i)[self.k-1]
            predictions = [a + ' ' + b + '\n' for a,b in zip(i, prediction)]
            for j in predictions:
                output.write(j)
            output.write('\n')

        output.close()



filepath = './ES/ES/train'
test_set = './ES/ES/dev.in'
t_prob = transition_MLE(filepath=filepath)
e_prob = emission_MLE(filepath=filepath)

viterbi = KBestViterbi(t_prob,e_prob,5)
viterbi.predict(test_set)