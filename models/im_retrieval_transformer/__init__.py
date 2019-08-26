import models.im_all_transformer_straight_attn as base
import models.im_all_transformer.input as base_input
from models.im_all_transformer.model import model_fn
from models.im_all_transformer import editor as base_edit
from models.im_retrieval_transformer.edit_encoder import EditEncoderAcc
from models.im_retrieval_transformer.input import parse_instance

NAME = 'im_retrieval_transformer'

base_edit.EditEncoder = EditEncoderAcc
base_input.parse_instance = parse_instance


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
