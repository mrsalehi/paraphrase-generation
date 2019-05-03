from models.common import util
from models.im_attn_ee_rnn_attn_dec_copy_net.input import parse_instance
from models.neural_editor.edit_noiser import EditNoiser


def create_formulas(plans, config):
    noiser = EditNoiser.from_config(config)
    free_set = util.get_free_words_set() if config.get('editor.use_free_set', False) else None

    formulas = []
    formula2plan = []
    for i, (base, edits) in enumerate(plans):
        for j, (src, tgt) in enumerate(edits):
            instance = {
                'base': base,
                'src': src,
                'tgt': tgt
            }
            formula = parse_instance(instance, noiser, free_set)

            formulas.append(formula)
            formula2plan.append((i, j))

    return formulas, formula2plan
