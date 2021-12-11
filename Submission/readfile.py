def read_train_file(filename):
    with open(filename, encoding='utf-8') as f:
        file_content = f.read()

    # Split the entire file into sentences. Output: List of sentences
    sentences = file_content.strip().split('\n\n')

    # Split each sentence into their token_tag pair
    # Output: List of sentences. Each sentence is a list of token_tag_pair
    token_tag_pairs = [i.split('\n') for i in sentences]

    # Separate each token_tag_pair into a list of [token, tag].
    # Output: [[[token, tag], [token, tag], ...], [[token, tag], [token, tag], ...], ...]
    for idx, sentence in enumerate(token_tag_pairs):
        token_tags = [i.rsplit(' ', maxsplit=1) for i in sentence]
        token_tag_pairs[idx] = token_tags

    return token_tag_pairs


def read_dev_in_file(filename):
    with open(filename, encoding='utf-8') as f:
        file_content = f.read()

    # Split the entire file into sentences. Output: List of sentences
    sentences = file_content.strip().split('\n\n')

    # Split each sentence into their tokens
    # Output: List of sentences. Each sentence is a list of tokens
    tokens = [i.split('\n') for i in sentences]

    return tokens