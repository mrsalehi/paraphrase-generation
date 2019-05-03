import models.neural_editor as base
import models.neural_editor.decoder as base_decoder
import models.neural_editor.input as base_input
import models.neural_editor.paraphrase_gen as base_para_gen
from models.im_attn_ee_rnn_attn_dec_copy_net.decoder import create_decoder_cell
from models.im_attn_ee_rnn_attn_dec_copy_net.input import read_examples_from_file, input_fn, input_fn_from_gen_multi
from models.im_attn_ee_rnn_attn_dec_copy_net.model import model_fn
from models.im_attn_ee_rnn_attn_dec_copy_net.paraphrase_gen import create_formulas

NAME = 'im_attn_ee_rnn_attn_dec_copy_net'

base_decoder.create_decoder_cell = create_decoder_cell
base_input.read_examples_from_file = read_examples_from_file
base_input.input_fn = input_fn
base_para_gen.create_formulas = create_formulas
base_para_gen.input_fn_from_gen_multi = input_fn_from_gen_multi


# def train(config, data_dir):
#     V, embed_matrix = vocab.read_word_embeddings(
#         data_dir / 'word_vectors' / config.editor.wvec_path,
#         config.editor.word_dim,
#         config.editor.vocab_size
#     )
#
#     estimator = base.get_estimator(config, embed_matrix, model_fn)
#
#     if config.get('eval.enable', True):
#         hooks = [
#             base.get_eval_hook(estimator,
#                                lambda: base_input.eval_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V)),
#                                name='eval',
#                                every_n_steps=config.eval.eval_steps),
#         ]
#     else:
#         hooks = []
#
#     lms_hook = LMSSessionRunHook({'optimization'})
#     hooks.append(lms_hook)
#
#     return estimator.train(
#         input_fn=lambda: base_input.train_input_fn(config, data_dir, vocab.create_vocab_lookup_tables(V)),
#         hooks=hooks,
#         max_steps=config.optim.max_iters
#     )

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
