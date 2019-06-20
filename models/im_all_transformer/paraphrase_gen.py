from tqdm import tqdm

from models.common import vocab, util
from models.im_all_transformer.input import parse_instance, map_word_to_sub_words, map_str_to_bytes, \
    input_fn_from_gen_multi
from models.neural_editor.edit_noiser import EditNoiser
from models.neural_editor.paraphrase_gen import read_plan


def create_formulas(plans, config):
    noiser = EditNoiser.from_config(config)
    free_set = util.get_free_words_set() if config.get('editor.use_free_set', False) else None

    formulas = []
    formula2plan = []
    for i, (base, edits) in enumerate(plans):
        output = ''
        basic_formula = (base, output)
        for j, edit_vector_pair in enumerate(edits):
            instance = basic_formula + edit_vector_pair
            instance = parse_instance('\t'.join(instance), noiser, free_set)

            if config.editor.get('use_sub_words', False):
                bpemb = vocab.get_bpemb_instance(config)
                instance = map_word_to_sub_words(instance, bpemb)

            formula = map_str_to_bytes(instance)

            formulas.append(formula)
            formula2plan.append((i, j))

    return formulas, formula2plan


def clean_sentence(sent):
    return sent.replace('<stop>', '').strip()


def generate(estimator, plan_path, checkpoint_path, config, V):
    vocab.create_vocab_lookup_tables(V)

    batch_size = config.optim.batch_size
    beam_width = config.editor.decoder.beam_size

    plans = read_plan(plan_path)
    formulas, formula2plan = create_formulas(plans, config)

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

    for i, o in enumerate(tqdm(output, total=len(formulas))):
        # paraphrases = [clean_sentence(j.decode('utf8')) for j in o['joined']]
        paraphrases = [clean_sentence(j.decode('utf8')) for j in o['joined']]
        assert len(paraphrases) == beam_width

        plan_index, edit_index = formula2plan[i]
        plan2paraphrase[plan_index][edit_index] = paraphrases

        # if 'attns_weight_0' in o and 'attns_weight_1' in o:
        #     plan2attn_weight[plan_index][edit_index] = (o['attns_weight_0'], o['attns_weight_1'])
        #
        # if 'dec_alg_base' in o:
        #     plan2dec_base_attn_weight[plan_index][edit_index] = o['dec_alg_base']
        #
        # if 'dec_alg_mev' in o:
        #     plan2dec_mev_attn_weight[plan_index][edit_index] = o['dec_alg_mev']

    assert len(plans) == len(plan2paraphrase)

    return plan2paraphrase, plan2attn_weight, plan2dec_base_attn_weight, plan2dec_mev_attn_weight
