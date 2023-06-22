import regex as re

def load_data(path):
    with open(path) as f:
        data = f.read()
    return data.split("\n\n")

def parse_labels(labels):
    parsed_labels = []
    labels= labels.split('}, {')
    for i in labels:
        i = re.sub('{','',i)
        i = re.sub('}','',i)
        i = i.split(', ')
        parsed_labels.append(i)
    return parsed_labels