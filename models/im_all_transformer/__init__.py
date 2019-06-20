import models.neural_editor as base
import models.neural_editor.input as base_input
import models.neural_editor.paraphrase_gen as base_para_gen
from models.im_all_transformer import input as new_input
from models.im_all_transformer.model import model_fn
from models.im_all_transformer.paraphrase_gen import create_formulas, generate

NAME = 'im_all_transformer'

# Patch input functions
base_input.input_fn = new_input.input_fn
base_input.input_fn_from_gen_multi = new_input.input_fn_from_gen_multi

base.train_input_fn = new_input.train_input_fn
base.train_big_input_fn = new_input.train_big_input_fn
base.eval_input_fn = new_input.eval_input_fn
base.eval_big_input_fn = new_input.eval_big_input_fn

# Patch paraphrase generator
base_para_gen.generate = generate
base_para_gen.create_formulas = create_formulas
base_para_gen.input_fn_from_gen_multi = new_input.input_fn_from_gen_multi


def train(*args):
    return base.train(*args, my_model_fn=model_fn)


def eval(*args):
    return base.eval(*args, my_model_fn=model_fn)


def predict(*args):
    return base.predict(*args, my_model_fn=model_fn)


def augment_meta_test(*args):
    return base.augment_meta_test(*args, my_model_fn=model_fn)


def augment_debug(*args):
    return base.augment_debug(*args, my_model_fn=model_fn)


def generate_paraphrase(*args):
    return base.generate_paraphrase(*args, my_model_fn=model_fn)
