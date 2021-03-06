import itertools
import pickle

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
    vocab.create_vocab_lookup_tables(V)

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

    # plan2paraphrase = [[None for _ in range(num_edit_vectors)] for _ in range(len(plans))]
    plan2paraphrase = [[None for _ in evs] for b, evs in plans]
    plan2attn_weight = [[None for _ in evs] for b, evs in plans]
    plan2dec_base_attn_weight = [[None for _ in evs] for b, evs in plans]
    plan2dec_mev_attn_weight = [[None for _ in evs] for b, evs in plans]

    save_attentions = config.get("eval.save_attentions", False)

    for i, o in enumerate(tqdm(output, total=len(formulas))):
        paraphrases = [clean_sentence(j.decode('utf8')) for j in o['joined']]
        assert len(paraphrases) == beam_width

        plan_index, edit_index = formula2plan[i]
        plan2paraphrase[plan_index][edit_index] = paraphrases

        if not save_attentions:
            continue

        if 'attns_weight_0' in o and 'attns_weight_1' in o:
            plan2attn_weight[plan_index][edit_index] = (o['attns_weight_0'], o['attns_weight_1'])

        if 'dec_alg_base' in o:
            plan2dec_base_attn_weight[plan_index][edit_index] = o['dec_alg_base']

        if 'dec_alg_mev' in o:
            plan2dec_mev_attn_weight[plan_index][edit_index] = o['dec_alg_mev']

    assert len(plans) == len(plan2paraphrase)

    return plan2paraphrase, plan2attn_weight, plan2dec_base_attn_weight, plan2dec_mev_attn_weight


def save_attn_weights(attn_weights, name):
    with open(name, 'wb') as f:
        pickle.dump(attn_weights, f)


def flatten(plan2paras):
    try:
        flatten_plan2paras = [list(itertools.chain.from_iterable(para_lst)) for para_lst in plan2paras]
        return flatten_plan2paras
    except:
        return [[]]
