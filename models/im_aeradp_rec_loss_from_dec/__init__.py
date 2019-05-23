import models.im_pg_rec_loss_from_dec.model as base_model
import models.neural_editor as base
import models.neural_editor.decoder as base_decoder
import models.neural_editor.input as base_input
import models.neural_editor.optimizer as base_optimizer
import models.neural_editor.paraphrase_gen as base_para_gen
from models.im_aeradp_rec_loss.optimizer import loss
from models.im_aeradp_rec_loss_from_dec.editor import editor_train
from models.im_attn_ee_rnn_attn_dec_copy_net.paraphrase_gen import create_formulas
from models.im_attn_ee_rnn_attn_dec_pg.decoder import create_decoder_cell
from models.im_attn_ee_rnn_attn_dec_pg.input import read_examples_from_file, input_fn, input_fn_from_gen_multi
from models.im_pg_rec_loss_from_dec.model import model_fn

NAME = 'im_aeradp_rec_loss_from_dec'

base_model.editor.editor_train = editor_train
base_decoder.create_decoder_cell = create_decoder_cell
base_optimizer.loss = loss
base_input.read_examples_from_file = read_examples_from_file
base_input.input_fn = input_fn
base_para_gen.create_formulas = create_formulas
base_para_gen.input_fn_from_gen_multi = input_fn_from_gen_multi


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
