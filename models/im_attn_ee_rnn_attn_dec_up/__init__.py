import models.im_attn_ee_rnn_attn_dec.editor as base_editor
import models.neural_editor as base
import models.neural_editor.decoder as base_decoder
import models.neural_editor.input as base_input
from models.im_attn_ee_rnn_attn_dec.decoder import create_decoder_cell
from models.im_attn_ee_rnn_attn_dec.model import model_fn
from models.im_attn_ee_rnn_attn_dec_up.edit_encoder import attn_encoder
from models.im_vev.input import read_examples_from_file

NAME = 'im_attn_ee_rnn_attn_dec_up'

base_editor.attn_encoder = attn_encoder
base_decoder.create_decoder_cell = create_decoder_cell
base_input.read_examples_from_file = read_examples_from_file


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
