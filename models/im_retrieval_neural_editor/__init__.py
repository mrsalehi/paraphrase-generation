import models.im_vev.model as base_model
import models.neural_editor as base
import models.neural_editor.input as base_input
from models.im_retrieval_neural_editor.editor import editor_train
from models.im_vev.input import read_examples_from_file
from models.im_vev.model import model_fn

NAME = 'im_vev'

base_input.read_examples_from_file = read_examples_from_file
base_model.editor.editor_train = editor_train


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
