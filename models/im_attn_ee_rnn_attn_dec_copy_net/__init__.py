import models.neural_editor as base
import models.neural_editor.decoder as base_decoder
import models.neural_editor.input as base_input
from models.im_attn_ee_rnn_attn_dec_copy_net.decoder import create_decoder_cell
from models.im_attn_ee_rnn_attn_dec_copy_net.input import read_examples_from_file, input_fn
from models.im_attn_ee_rnn_attn_dec_copy_net.model import model_fn

NAME = 'im_attn_ee_rnn_attn_dec_copy_net'

base_decoder.create_decoder_cell = create_decoder_cell
base_input.read_examples_from_file = read_examples_from_file
base_input.input_fn = input_fn


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
