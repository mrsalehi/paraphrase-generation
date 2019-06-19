from models.common import vocab, util
from models.im_all_transformer.input import parse_instance, map_word_to_sub_words, map_str_to_bytes
from models.neural_editor.edit_noiser import EditNoiser


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
