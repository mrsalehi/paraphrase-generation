import models.neural_editor as base
from models.im_all_transformer_proc_loss.model import model_fn
from models.im_all_transformer import generate_paraphrase

NAME = 'im_all_transformer_proc_loss'


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


# def generate_paraphrase(*args):
#     return base.generate_paraphrase(*args, my_model_fn=model_fn)
