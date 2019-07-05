import models.im_all_transformer_straight_attn as base
from models.im_all_transformer_straight_attn import model_fn
from models.im_all_transformer import edit_encoder as base_edit_encoder
from models.im_all_transformer_straight_attn_rm_tok.edit_encoder import TransformerMicroEditExtractorWithRmTok

NAME = 'im_all_transformer_straight_attn_rm_tok'

# New MicroEditExtractor functions
base_edit_encoder.TransformerMicroEditExtractor = TransformerMicroEditExtractorWithRmTok


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
