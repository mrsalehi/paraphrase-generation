# import tensorflow as tf

# import models.im_attn_ee_rnn_attn_dec_copy_net.memory_saving_gradients as memory_saving_gradients

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

import fire

from models import im_all_transformer
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_all_transformer
    fire.Fire(cls)
