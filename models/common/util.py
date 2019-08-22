import subprocess
from collections import deque

import numpy as np


def get_num_total_lines(p):
    # result = subprocess.run(['wc', '-l', p], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if not isinstance(p, str):
        p = str(p)

    result = subprocess.check_output(['wc', '-l', p])
    if isinstance(result, bytes):
        result = result.decode('utf-8')

    return int(result.strip().split(' ')[0])


def jaccard(s1, s2):
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))


def read_tsv(filepath, parse_words=False):
    with open(filepath, encoding='utf8') as f:
        rows = map(lambda x: tuple(x.strip().lower().split('\t')), f)
        if parse_words:
            rows = map(lambda x: tuple(col.split(' ') for col in x), rows)

        rows = list(rows)

    return rows


def save_tsv(filepath, rows):
    with open(filepath, 'w', encoding='utf8') as f:
        for cols in rows:
            f.write('\t'.join([str(c) for c in cols]))
            f.write('\n')


def get_free_words_set():
    with open('free_words.txt', encoding='utf8') as f:
        free = set([l.strip() for l in f])

    return free


def load_str_list(filepath):
    lst = deque()
    with open(filepath, encoding='utf8') as f:
        for l in f:
            lst.append(l[:-1])

    return lst


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
