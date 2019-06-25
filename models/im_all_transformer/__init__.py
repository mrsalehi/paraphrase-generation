import models.neural_editor as base
import models.neural_editor.input as base_input
import models.neural_editor.paraphrase_gen as base_para_gen
from models.im_all_transformer import input as new_input
from models.im_all_transformer.model import model_fn
from models.im_all_transformer.paraphrase_gen import create_formulas, generate, save_outputs

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


def train(*args, my_model_fn=model_fn, **kwargs):
    return base.train(*args, my_model_fn=my_model_fn, **kwargs)


def eval(*args, my_model_fn=model_fn, **kwargs):
    return base.eval(*args, my_model_fn=my_model_fn, **kwargs)


def predict(*args, my_model_fn=model_fn, **kwargs):
    return base.predict(*args, my_model_fn=my_model_fn, **kwargs)


def augment_meta_test(*args, my_model_fn=model_fn, **kwargs):
    return base.augment_meta_test(*args, my_model_fn=my_model_fn, **kwargs)


def augment_debug(*args, my_model_fn=model_fn, **kwargs):
    return base.augment_debug(*args, my_model_fn=my_model_fn, **kwargs)


def generate_paraphrase(config, data_dir, checkpoint_path, plan_path, output_path, beam_width, batch_size,
                        my_model_fn=model_fn, **kwargs):
    V, embed_matrix = base.get_vocab_embedding_matrix(config, data_dir)

    if batch_size:
        config.put('optim.batch_size', batch_size)

    if beam_width:
        config.put('editor.beam_width', beam_width)

    estimator = base.get_estimator(config, embed_matrix, my_model_fn)

    outputs = generate(estimator, plan_path, checkpoint_path, config, V)
    save_outputs(outputs, output_path)
