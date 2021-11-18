import pprint
import pandas as pd

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
            token, tag = line.split(sep=separator)

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
    print(t_prob)

    return transition_probability

filepath = './ES/ES/train'
transition_MLE(filepath=filepath)    