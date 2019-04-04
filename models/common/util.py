import subprocess


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
