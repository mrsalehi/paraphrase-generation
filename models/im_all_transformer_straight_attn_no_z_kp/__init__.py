import models.im_all_transformer_straight_attn as base
from models.im_all_transformer_straight_attn import model_fn
from models.im_all_transformer import editor as base_editor
from models.im_all_transformer_straight_attn_no_z_kp.edit_encoder import EditEncoderNoZ
from models.im_all_transformer_straight_attn_no_z_kp.decoder import Decoder

NAME = 'im_all_transformer_straight_attn_no_z_kp'

base_editor.EditEncoder = EditEncoderNoZ
base_editor.Decoder = Decoder


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
