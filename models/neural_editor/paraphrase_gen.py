import itertools

from tqdm import tqdm

from models.common import vocab, util
from models.common.util import read_tsv
from models.neural_editor import convert_to_bytes, parse_instance, input_fn_from_gen_multi
from models.neural_editor.edit_noiser import EditNoiser


def read_plan(src_path):
    rows = read_tsv(src_path)
    plans = []

    for r in rows:
        base, edits = r[0], r[1:]
        assert len(edits) % 2 == 0

        edit_vector_pairs = [(edits[i], edits[i + 1]) for i in range(len(edits))[::2]]
        plans.append((base, edit_vector_pairs))

    return plans


def create_formulas(plans, config):
    noiser = EditNoiser.from_config(config)
    free_set = util.get_free_words_set() if config.get('editor.use_free_set', False) else None

    formulas = []
    formula2plan = []
    for i, (base, edits) in enumerate(plans):
        for j, edit_vector_pair in enumerate(edits):
            base_words = convert_to_bytes(base.split(' ')),
            edit_instance = parse_instance(edit_vector_pair, noiser, free_set)
            formula = base_words + edit_instance

            formulas.append(formula)
            formula2plan.append((i, j))

    return formulas, formula2plan


def clean_sentence(sent):
    return sent.replace('<stop>', '').strip()


def generate(estimator, plan_path, checkpoint_path, config, V):
    batch_size = config.optim.batch_size
    if config.get('editor.use_beam_decoder', False):
        beam_width = config.editor.beam_width
    else:
        beam_width = 1

    plans = read_plan(plan_path)
    formulas, formula2plan = create_formulas(plans, config)

    num_edit_vectors = int(len(formulas) / len(plans))

    formula_gen = lambda: iter(formulas)
    output = estimator.predict(
        input_fn=lambda: input_fn_from_gen_multi(formula_gen, vocab.create_vocab_lookup_tables(V), batch_size),
        checkpoint_path=checkpoint_path
    )

    plan2paraphrase = [[None for _ in range(num_edit_vectors)] for _ in range(len(plans))]

    for i, o in enumerate(tqdm(output, total=len(formulas))):
        paraphrases = [clean_sentence(j.decode('utf8')) for j in o['joined']]
        assert len(paraphrases) == beam_width

        plan_index, edit_index = formula2plan[i]
        plan2paraphrase[plan_index][edit_index] = paraphrases

    assert len(plans) == len(plan2paraphrase)

    return plan2paraphrase


def flatten(plan2paras):
    flatten_plan2paras = [list(itertools.chain.from_iterable(para_lst)) for para_lst in plan2paras]
    return flatten_plan2paras
