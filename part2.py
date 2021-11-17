def est_transition(filepath):
    file = open(filepath, 'r', encoding='UTF-8')
    separator = ' '

    tag_count = {
        'O': 0,
        'B-positive': 0, 
        'B-neutral': 0, 
        'B-negative': 0, 
        'I-positive': 0, 
        'I-neutral': 0, 
        'I-negative': 0,
        'START' : 0,
        'STOP' : 0
    }
 
    transition_count = {
        'O': {},
        'B-positive': {}, 
        'B-neutral': {}, 
        'B-negative': {}, 
        'I-positive': {}, 
        'I-neutral': {}, 
        'I-negative': {},
        'START' : {},
        'STOP' : {}
    }

    prev_tag = 'START'
    for line in file:
        if line == '\n':
            if prev_tag != "START":
                tag_count['START'] += 1
                tag_count['STOP'] += 1
                if "STOP" not in transition_count[prev_tag]:
                    transition_count[prev_tag]["STOP"] = 1
                else:
                    transition_count[prev_tag]["STOP"] += 1
                prev_tag = 'START'
            continue
        else:
            # separate each line into their token and tags
            line = line.strip()
            token, tag = line.split(sep=separator)

            # count the total number of tag occurrences
            tag_count[tag] += 1
            
            if tag not in transition_count[prev_tag]:
                transition_count[prev_tag][tag] = 1
            else:
                transition_count[prev_tag][tag] += 1
            prev_tag = tag

    transition_para = transition_count
    for key1, value1 in transition_para.items():
        for key2, value2 in value1.items():
            transition_para[key1][key2] = value2/tag_count[key1]
    print(tag_count)
    print(transition_count)
    return transition_para

print(est_transition('./ES/ES/train'))
