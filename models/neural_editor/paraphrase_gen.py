import itertools

from tqdm import tqdm

from models.common import vocab
from models.common.util import read_tsv
from models.neural_editor import convert_to_bytes, parse_instance, input_fn_from_gen_multi


class EditVectorPair:
    def __init__(self, src, tgt, paraphrase=None):
        self.src = src
        self.tgt = tgt
        self.paraphrase = paraphrase


class Runner:
    def __init__(self, exp_dir, checkpoint=None, output=None):
        self.ckpt_num = checkpoint
        self.exp_dir = exp_dir
        self.output = output

    def generate(self, src):
        pass


def read_plan(src_path):
    rows = read_tsv(src_path)
    plans = []

    for r in rows:
        base, edits = r[0], r[1:]
        assert len(edits) % 2 == 0

        edit_vector_pairs = [(edits[i], edits[i + 1]) for i in range(len(edits))[::2]]
        plans.append((base, edit_vector_pairs))

    return plans


def create_formulas(plans):
    formulas = []
    formula2plan = []
    for i in range(len(plans)):
        p = plans[i]
        base = p[0]

        for j, edit_vector_pair in enumerate(p[1]):
            base_words = convert_to_bytes(base.split(' ')),
            edit_instance = parse_instance(edit_vector_pair)
            formula = base_words + edit_instance

            formulas.append(formula)
            formula2plan.append((i, j))

    return formulas, formula2plan


def generate(estimator, plan_path, checkpoint_path, batch_size, beam_width, V):
    plans = read_plan(plan_path)
    formulas, formula2plan = create_formulas(plans)

    num_edit_vectors = int(len(formulas) / len(plans))

    output = estimator.predict(
        input_fn=lambda: input_fn_from_gen_multi(iter(formulas), vocab.create_vocab_lookup_tables(V), batch_size),
        checkpoint_path=checkpoint_path
    )

    plan2paraphrase = [[None] * num_edit_vectors] * len(plans)

    for i, o in enumerate(tqdm(output, total=len(formulas))):
        paraphrases = [j.decode('utf8') for j in o['joined']]
        assert len(paraphrases) == beam_width

        plan_index, edit_index = formula2plan[i]
        plan2paraphrase[plan_index][edit_index] = paraphrases

    assert len(plans) == len(plan2paraphrase)

    return plan2paraphrase


def flatten(plan2paras):
    flatten_plan2paras = [list(itertools.chain.from_iterable(para_lst)) for para_lst in plan2paras]
    return flatten_plan2paras
