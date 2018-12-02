import sys
from os.path import join
from shutil import copyfile


def filter_len(x):
    x = x[:-1]
    src, tgt = x.split('\t')

    src = src.split(' ')
    tgt = tgt.split(' ')

    return len(src) <= 40 and len(tgt) <= 40


def fix(inp, out):
    with open(out, 'w', encoding='utf8') as fout:
        with open(inp, 'r', encoding='utf8') as fin:
            fixed = filter(filter_len, fin)
            for i in fixed:
                fout.write(i)


def fix_all(dataset):
    copyfile(join(dataset, 'train.tsv'), join(dataset, 'train.big.tsv'))
    copyfile(join(dataset, 'valid.tsv'), join(dataset, 'valid.big.tsv'))

    fix(join(dataset, 'train.big.tsv'), join(dataset, 'train.tsv'))
    fix(join(dataset, 'valid.big.tsv'), join(dataset, 'valid.tsv'))


if __name__ == '__main__':
    fix_all(sys.argv[1])
