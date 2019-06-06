import models.im_attn_ee_rnn_attn_dec_pg.decoder as base_decoder
import models.neural_editor as base
import models.neural_editor.optimizer as base_optimizer
import models.neural_editor.paraphrase_gen as base_para_gen
from models.im_aeradp_no_mev_attn.decoder import create_decoder_cell
from models.im_attn_ee_rnn_attn_dec_copy_net.paraphrase_gen import create_formulas
from models.im_attn_ee_rnn_attn_dec_pg.model import model_fn
from models.im_attn_ee_rnn_attn_dec_pg.optimizer import loss

NAME = 'im_aeradp_no_mev_attn'

base_decoder.create_decoder_cell = create_decoder_cell
base_optimizer.loss = loss
base_para_gen.create_formulas = create_formulas


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
