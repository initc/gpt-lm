
import json
import sys
import codecs
from nltk.translate.bleu_score import sentence_bleu


def read_file(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    content = []
    with codecs.open(file, 'r', 'utf-8') as f:
        for l in f:
            l = json.loads(l)
            content.append(l)
    return content


def eval_bleu(data):
    sizes = len(data)
    bleu_1 = 0
    for d in data:
        references = d["references"]
        response = d["response"]
        score = sentence_bleu(references, response, weights=(1, 0, 0, 0))
        bleu_1 += score
    bleu_2 = 0
    for d in data:
        references = d["references"]
        response = d["response"]
        score = sentence_bleu(references, response, weights=(0.5, 0.5, 0, 0))
        bleu_2 += score
    return bleu_1/sizes, bleu_2/sizes
    

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Too few args for this script')

    generater_file = sys.argv[1]
    responses = read_file(generater_file)

    bleu_1, bleu_2 = eval_bleu(responses)
    print('| BLEU 1 : {}, BLEU 2 : {}'.format(bleu_1, bleu_2))