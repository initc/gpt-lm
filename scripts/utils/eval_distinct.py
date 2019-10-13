
import json
import sys
import codecs


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
            content.append(l["response"])
    return content


def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)


def eval_distinct(hyps_resp):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    hyps_resp = [list(i) for i in hyps_resp]
    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    num_tokens = sum([len(i) for i in hyps_resp])
    dist1 = count_ngram(hyps_resp, 1) / float(num_tokens)
    dist2 = count_ngram(hyps_resp, 2) / float(num_tokens)

    return (dist1 + dist2) / 2.0


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Too few args for this script')

    generater_file = sys.argv[1]
    generater_file = read_file(generater_file)

    distinct = eval_distinct(generater_file)
    print('| Distinct ', distinct)
