import sys
import pprint

# Helper function to check if the token is part of the emission probability table
def check_token(token,dictionary):
    for key, value in dictionary.items():
        if token in value:
            return True
    return False


def emission_MLE(filepath):
    file = open(filepath, 'r', encoding='UTF-8')
    separator = ' '
    k = 1
    unknown_token = '#UNK#'

    # states = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']
    tag_count = {
        'O': 0,
        'B-positive': 0, 
        'B-neutral': 0, 
        'B-negative': 0, 
        'I-positive': 0, 
        'I-neutral': 0, 
        'I-negative': 0
        }

    emission_count = {}
    for line in file:
        if line == '\n':
            continue
        else:
            # separate each line into their token and tags
            line = line.strip()
            token, tag = line.split(sep=separator)

            # count the total number of tag occurrences
            tag_count[tag] += 1

            # create new key value pair in emission_count per tag
            if tag not in emission_count:
                emission_count[tag] = dict()

            # count the number of occurrences from tag to token
            if token not in emission_count[tag]:
                emission_count[tag][token] = 1
            else:
                emission_count[tag][token] += 1

    # create a copy of the emission_count dictionary
    emission_probability = emission_count

    # calculate the emission probability by dividing occurrence by total count
    for key, value in emission_probability.items():
        # Naive way to handle tokens in test set that do not appear in training set
        value[unknown_token] = k/(tag_count[key]+k)

        for key2, value2 in value.items():
            value[key2] /= (tag_count[key]+k)
    
    # pprint.pprint(emission_probability)
    return emission_probability


def predict_tag(test_set, training_set):

    file = open(test_set, 'r', encoding='UTF-8')
    output = open(test_set + '/../dev.p1.out', 'w', encoding='UTF-8')

    unknown_token = '#UNK#'

    emission_probability = emission_MLE(filepath=training_set)

    for line in file:
        if line == '\n':
            output.write(line)
            continue
        else:
            line = line.strip()

            predicted_tag_probability = 0
            predicted_tag = None

            # Check if token is in emission probability table
            if check_token(line, emission_probability):
                for key, value in emission_probability.items():
                    if line in value:
                    # Update max probability and predicted tag
                        if value[line] > predicted_tag_probability:
                            predicted_tag_probability = value[line]
                            predicted_tag = key
            else:
                for key, value in emission_probability.items():
                    # Update max probability and predicted tag
                    if value[unknown_token] > predicted_tag_probability:
                        predicted_tag_probability = value[unknown_token]
                        predicted_tag = key

        output.write(line + ' ' + predicted_tag + '\n')
    output.close()
    return


# ACTUAL CODE TO RUN
'''
To use this code,
1) Run "python part1.py ./<DIRECTORY>/<DIRECTORY>/dev.in ./<DIRECTORY>/<DIRECTORY>/train".
2) Make sure to change <DIRECTORY> to the specific directory of interest - Either ES or RU. 

**** Please make sure you have installed Python 3.4 or above.
**** On Windows, you can run "python part1.py ./<DIRECTORY>/<DIRECTORY>/dev.in ./<DIRECTORY>/<DIRECTORY>/train"
**** On Linux or Mac, you need to run "python3 part1.py ./<DIRECTORY>/<DIRECTORY>/dev.in ./<DIRECTORY>/<DIRECTORY>/train"

'''

# if len(sys.argv) < 3:
#     print ('Please make sure you have installed Python 3.4 or above!')
#     print ("Usage on Windows:  python evalResult.py gold predictions")
#     print ("Usage on Linux/Mac:  python3 evalResult.py gold predictions")
#     sys.exit()

# test_set = sys.argv[1]
# training_set = sys.argv[2]

# predict_tag(test_set, training_set)
