import numpy as np
import pprint

class HMM:

    def __init__(self, k = 1):
        self.possible_states = []

        self.emission_count = {}
        self.transition_count = {}
        self.state_count = {}

        self.k = k

        self.predictions = None

        self.emission_probability = {}
        self.transition_probability = {}
        pass


    def _count_all(self):
        self.transition_count['START'] = {}
        for sentence in self.train_dataset:
            prev_state = None

            for token, tag in sentence:
                if tag not in self.possible_states:
                    self.possible_states.append(tag)
                
                if self.emission_count.get(token) == None:
                    self.emission_count[token] = {}
                self.emission_count[token][tag] = self.emission_count[token].get(tag, 0) + 1

                if prev_state != None:
                    if self.transition_count.get(prev_state) == None:
                        self.transition_count[prev_state] = {}
                    self.transition_count[prev_state][tag] = self.transition_count[prev_state].get(tag, 0) + 1
                
                else:
                    self.transition_count['START'][tag] = self.transition_count['START'].get(tag, 0) + 1
                    self.state_count['START'] = self.state_count.get('START', 0) + 1
                
                self.state_count[tag] = self.state_count.get(tag, 0) + 1
                prev_state = tag

            self.transition_count[prev_state]['STOP'] = self.transition_count[prev_state].get('STOP', 0) + 1

        return self


    def _calculate_emission_MLE_UNK(self, x, y):
        if self.emission_count.get(x) == None:
            x = '#UNK#'
        
        self.emission_probability[x] = self.emission_probability.get(x, {})
        if self.emission_probability[x].get(y) != None:
            return self.emission_probability[x][y]

        if x == '#UNK#':
            self.emission_probability[x][y] = self.k / (self.state_count.get(y, 0) + self.k)
        else:
            self.emission_probability[x][y] = self.emission_count.get(x, {}).get(y, 0) / (self.state_count[y] + self.k)
        
        return self.emission_probability[x][y]


    def _get_argmax_y_part1(self, x):
        probabilities = {}

        for state in self.possible_states:
            probabilities[state] = self._calculate_emission_MLE_UNK(x, state)

        return max(probabilities, key=probabilities.get)


    def _calculate_transition_MLE(self, prev_tag, tag):
        self.transition_probability[prev_tag] = self.transition_probability.get(prev_tag, {})

        if self.transition_probability[prev_tag].get(tag) == None:
            self.transition_probability[prev_tag][tag] = self.transition_count[prev_tag].get(tag, 0) / self.state_count.get(prev_tag)

        return self.transition_probability[prev_tag][tag]


    def _viterbi(self, sentence):
        # BASE CASE
        scores = {
            0: {
                'START' : 0
            }
        }

        index = 1

        # Forward Algorithm - From START to index N
        for token in sentence:
            scores[index] = {}

            for state in self.possible_states:
                state_scores = {}

                for prev_tag in scores[index-1].keys():
                    t_prob = self._calculate_transition_MLE(prev_tag, state)
                    e_prob = self._calculate_emission_MLE_UNK(token, state)

                    if t_prob > 0 and e_prob > 0:
                        state_scores[prev_tag] = \
                            scores[index-1][prev_tag] + \
                            np.log(t_prob) + \
                            np.log(e_prob)
                    else:
                        state_scores[prev_tag] = float('-inf')

                best_score = max(state_scores.values())
                scores[index][state] = best_score

            index += 1

        # Forward Algorithm - From index N to STOP
        state_scores = {}
        for prev_tag in scores[index-1].keys():
            t_prob = self._calculate_transition_MLE(prev_tag, 'STOP')
            if t_prob > 0:
                state_scores[prev_tag] = scores[index-1][prev_tag] + np.log(t_prob)
            else:
                state_scores[prev_tag] = float('-inf')

        y_n = max(state_scores, key=state_scores.get)
        prediction_reversed = [y_n]

        # Backtracking Algorithm
        for n in reversed(range(1,index)):
            state_scores = {}

            for state in scores[n-1].keys():
                t_prob = self._calculate_transition_MLE(state, prediction_reversed[-1])

                if t_prob > 0:
                    state_scores[state] = scores[n-1][state] + np.log(t_prob)

            if all(prob == float('-inf') for prob in state_scores.values()):
                prediction_reversed.append('O')
            else:
                best_state = max(state_scores, key=state_scores.get)
                prediction_reversed.append(best_state)

        prediction = []
        prediction_reversed.reverse()

        for idx, token in enumerate(sentence):
            prediction.append([token, prediction_reversed[idx+1]])

        return prediction


    def _k_best_viterbi(self, sentence, k_num):
        # BASE CASE
        scores = {
            0: {
                'START' : [[1,[]]]
            }
        }

        index = 1

        # Forward Algorithm - From START to index N
        for token in sentence:
            scores[index] = {}

            for state in self.possible_states:
                state_scores = {}

                for prev_tag in scores[index-1].keys():
                    t_prob = self._calculate_transition_MLE(prev_tag, state)
                    e_prob = self._calculate_emission_MLE_UNK(token, state)
                    
                    score_paths = scores[index-1][prev_tag]

                    if t_prob > 0 and e_prob > 0:
                        for i in range(len(score_paths)):
                            state_scores[tuple(score_paths[i][1] + [prev_tag])] = \
                                score_paths[i][0] + \
                                np.log(t_prob) + \
                                np.log(e_prob)
                    else:
                        for i in range(len(score_paths)):
                            state_scores[tuple(score_paths[i][1] + [prev_tag])] = float('-inf')

                state_scores_V = sorted(state_scores.values(), reverse=True)
                state_scores_K = sorted(state_scores, key=state_scores.get, reverse=True)

                if len(state_scores_V) == 1:
                    scores[index][state] = [[state_scores_V[0],list(state_scores_K[0])]]
                else:
                    scores[index][state] = [[state_scores_V[idx],list(state_scores_K[idx])] for idx in range(k_num)]

            index += 1

        # Forward Algorithm - From index N to STOP
        state_scores = {}
        for prev_tag in scores[index-1].keys():
            t_prob = self._calculate_transition_MLE(prev_tag, 'STOP')
            score_paths = scores[index-1][prev_tag]
            if t_prob > 0:
                for i in range(len(score_paths)):
                    state_scores[tuple(score_paths[i][1] + [prev_tag])] = score_paths[i][0] + np.log(t_prob)
            else:
                for i in range(len(score_paths)):
                    state_scores[tuple(score_paths[i][1] + [prev_tag])] = float('-inf')

        state_scores_V = sorted(state_scores.values(), reverse=True)
        state_scores_K = sorted(state_scores, key=state_scores.get, reverse=True)

        k_best_path = state_scores_K[k_num-1]

        prediction = []

        for idx, token in enumerate(sentence):
            prediction.append([token, k_best_path[idx+1]])

        return prediction


    def predict_part1(self, test_dataset):
        '''
        self.predictions = 
        [
            [[token 1, prediction 1], [token 2, prediction 2], ...],
            [[token 1, prediction 1], [token 2, prediction 2], ...],
        ]
        '''
        self.predictions = []

        for sentence in test_dataset:
            sentence_pred = []

            for token in sentence:
                prediction = self._get_argmax_y_part1(token)
                sentence_pred.append([token, prediction])

            self.predictions.append(sentence_pred)

            # print(f'self.pred = {self.predictions[0]}')

        return self

    
    def predict_part2(self, test_dataset):
        self.predictions = []
        for sentence in test_dataset:
            sentence_pred = self._viterbi(sentence)
            self.predictions.append(sentence_pred)
        return self


    def predict_part3(self, test_dataset, k_num=5):
        self.predictions = []
        for sentence in test_dataset:
            sentence_pred = self._k_best_viterbi(sentence, k_num)
            self.predictions.append(sentence_pred)
        return self


    def train(self, train_dataset):
        self.train_dataset = train_dataset
        self._count_all()
        return self


    def write_preds(self, filename):
        output = ""
        for sentence in self.predictions:
            for token_tag_pair in sentence:
                output += "{}\n".format(" ".join(token_tag_pair))
            output += "\n"
        output += "\n"

        with open(filename, "w", encoding='utf-8') as f:
            f.write(output)
        return self