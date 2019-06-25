import models.im_all_transformer_shr_enc as base
import models.im_all_transformer_shr_enc.edit_encoder as base_edit_encoder
from models.im_all_transformer_shr_enc_no_grad.edit_encoder import TransformerMicroEditExtractor
from models.im_all_transformer_straight_attn import model_fn

NAME = 'im_all_transformer_shr_enc_no_grad'

base_edit_encoder.TransformerMicroEditExtractor = TransformerMicroEditExtractor


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
