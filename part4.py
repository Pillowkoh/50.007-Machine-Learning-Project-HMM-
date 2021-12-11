from collections import Counter, defaultdict
import time
import re
import string
import pprint

class StructuredPerceptron:
    def __init__(self):
        """ Class implementing Structured Perceptron model.
            Attributes:
                possible_states: List of possible states. E.g. 
                    [ "state_1", ..., "state_n" ]
                predictions: List of predictions made. E.g.
                    [
                        [
                            ["this", "state1"],
                            ["is", "state2"],
                            ["an", "state3"],
                            ["example", "state4"],
                        ],
                        ...
                    ]
                feature_weights: Dictionary containing individual feature weights.
                total_feature_weights: Dictionary containing overall feature weights.
        """
        self.possible_states = []

        self.predictions = None

        self.feature_weights = defaultdict(float)
        self.total_feature_weights = defaultdict(tuple)
    
    def _count_all(self):
        """ Count possible states. 
            Returns: 
                self.
        """
        states = set()
        for sentence in self.train_dataset:
            for _, tag in sentence:
                states.add(tag)
        self.possible_states = list(states)
        return self

    def train(self, train_dataset, no_of_epochs=10, learning_rate=0.2):
        """ Train model.
            Args: 
                train_data: list of parsed training data of the following form:
                    [
                        [
                            ["this", "state1"],
                            ["is", "state2"],
                            ["an", "state3"],
                            ["example", "state4"],
                        ],
                        ...
                    ]
            
            Returns:
                self.
        """
        assert no_of_epochs > 0

        self.train_dataset = train_dataset
        self._count_all()

        for epoch in range(no_of_epochs):
            print(f"Training epoch {epoch+1} with learning rate {learning_rate}...")
            total = correct = 0
            start = time.time()
            
            for labelled_sentence in train_dataset:
                tokens = [token_tag_pair[0] for token_tag_pair in labelled_sentence]
                tags = [token_tag_pair[1] for token_tag_pair in labelled_sentence]

                predicted_tags = self._viterbi(tokens)

                gold_features = self._get_global_features(tokens, tags)
                prediction_features = self._get_global_features(tokens, predicted_tags)

                if predicted_tags != tags:
                    self._update(gold_features, learning_rate, epoch+1)
                    for feature, count in prediction_features.items():
                        self.feature_weights[feature] = self.feature_weights[feature] - learning_rate * count
                
                total += len(labelled_sentence)
                correct += sum([1 for (predicted, gold) in zip(predicted_tags, tags) if predicted == gold])

            print(f"Training accuracy: {correct/total}")
            end = time.time()
            print(f"Time taken for {epoch + 1}th iteration: {end - start} seconds")
            print('-'*50)
            
        return self

    def _viterbi(self, example):
        """ Implementation of viterbi algorithm.
            Args:
                example: List of strings for words in example.
            
            Returns:
                List of predictions for example.
        """
        # Base Case
        k = 1
        pi = {
            k: {}
        }
        pi_edge = {
            k: {}
        }
        for state in self.possible_states:
            features = self._get_features(example[0], state, "START")
            feature_weights = sum((self.feature_weights[x] for x in features))
            # print(f'feature weights: {tuple(self.feature_weights[x] for x in features)}')
            pi[k][state] = feature_weights
            pi_edge[k][state] = "START"

        # Move forward recursively
        for observation in example[1:]:
            k += 1
            pi[k] = {}
            pi_edge[k] = {}
            for v in self.possible_states:
                probabilities = {}
                for u in pi[k-1].keys():
                    features = self._get_features(observation, v, u)
                    feature_weights = sum((self.feature_weights[x] for x in features))

                    probabilities[u] = pi[k-1][u] + feature_weights
                    
                max_state = max(probabilities, key=probabilities.get)
                pi[k][v] = probabilities[max_state]
                pi_edge[k][v] = max_state
            
        
        # Transition to STOP
        probabilities = {}
        for u in pi[k].keys():
            feature_weights = self.feature_weights[(u, "STOP")]
            probabilities[u] = pi[k][u] + feature_weights
        
        # Best y_n
        y_n = max(probabilities, key=probabilities.get)
        state_pred_r = [y_n]

        # Backtrack
        for n in reversed(range(1, k+1)):
            next_state = state_pred_r[-1]
            state_pred_r.append(pi_edge[n][next_state])
        state_pred_r.reverse()
        
        return state_pred_r[1:]

    def _get_global_features(self, tokens, tags):
        feature_counts = Counter()
        for idx, (token, tag) in enumerate(zip(tokens, tags)):
            if idx == 0:
                prev_tag = "START" 
            else:
                prev_tag = tags[idx-1]
            feature_counts.update(self._get_features(token, tag, prev_tag))
            # pprint.pprint(feature_counts)
        return feature_counts

    def _get_features(self, token, tag, prev_tag):
        word_lower = token.lower()
        prefix3 = word_lower[:3]
        prefix2 = word_lower[:2]
        suffix3 = word_lower[-3:]
        suffix2 = word_lower[-2:]

        features = [
            f'PREFIX2_{prefix2}',
            f'PREFIX2+TAG_{prefix2}_{tag}',
            f'PREFIX2+2TAGS_{prefix2}_{prev_tag}_{tag}',
            f'PREFIX3_{prefix3}',
            f'PREFIX3+TAG_{prefix3}_{tag}',
            f'PREFIX3+2TAGS_{prefix3}_{prev_tag}_{tag}',
            f'SUFFIX2_{suffix2}',
            f'SUFFIX2+TAG_{suffix2}_{tag}',
            f'SUFFIX2+2TAGS_{suffix2}_{prev_tag}_{tag}',
            f'SUFFIX3_{suffix3}',
            f'SUFFIX3+TAG_{suffix3}_{tag}',
            f'SUFFIX3+2TAGS_{suffix3}_{prev_tag}_{tag}',
            f'WORD_LOWER+TAG_{word_lower}_{tag}',
            f'WORD_LOWER+TAG_BIGRAM_{word_lower}_{tag}_{prev_tag}',
            f'UPPER_{token[0].isupper()}_{tag}',
            f'TAG_{tag}',
            f'TAG_BIGRAM_{prev_tag}_{tag}',
            f'DASH_{"-" in token}_{tag}',
            f'WORDSHAPE_{self._shape(token)}_TAG_{tag}',
            f'ISPUNC_{token in string.punctuation}'
        ]
        
        return features
    
    def _update(self, features, learning_rate, epoch_no):
        """ Updates feature weights. 
        
            Returns:
                self.
        """
        for f, count in features.items():
            w = self.feature_weights[f]

            # if self.total_feature_weights[f] is an empty tuple, initialise w_iter, total_weight to (0,0)
            if not self.total_feature_weights[f]:
                w_iteration, total_weight = (0, 0)
            else:
                w_iteration, total_weight = self.total_feature_weights[f]
            # Update weight sum with last registered weight since it was updated
            total_weight += (epoch_no - w_iteration) * w
            w_iteration = epoch_no
            total_weight += learning_rate * count

            # Update weight and total
            self.feature_weights[f] += learning_rate * count
            self.total_feature_weights[f] = (w_iteration, total_weight)
        
        return self
    
    def _shape(self, word):
        """ Get 'shape' of word (caps, numbers, etc.).
            Returns:
                String describing shape of word.
        """
        result = []
        for char in word:
            if char.isupper():
                result.append('X')
            elif char.islower():
                result.append('x')
            elif char in '0123456789':
                result.append('d')
            else:
                result.append(char)
        return re.sub(r"x+", "x*", ''.join(result))

    def predict(self, test_dataset):
        self.predictions = []
        for sentence in test_dataset:
            sentence_pred = self._viterbi(sentence)
            self.predictions.append(list(zip(sentence, sentence_pred)))
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
