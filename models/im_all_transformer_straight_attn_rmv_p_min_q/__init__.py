import models.im_all_transformer.decoder as base_decoder
import models.im_all_transformer as base
from models.im_all_transformer import edit_encoder as base_edit_encoder
from models.im_all_transformer.model import model_fn
from models.im_all_transformer_straight_attn_rmv_p_min_q.edit_encoder import EditEncoderRemovePMinusQ
from models.im_all_transformer_straight_attn.decoder import StraightAttentionDecoderStack

NAME = 'im_all_transformer_straight_attn_rmv_p_min_q'

# New Decoder functions
base_edit_encoder.EditEncoder = EditEncoderRemovePMinusQ


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


def generate_paraphrase(*args, my_model_fn=model_fn, **kwargs):
    return base.generate_paraphrase(*args, my_model_fn=my_model_fn, **kwargs)
