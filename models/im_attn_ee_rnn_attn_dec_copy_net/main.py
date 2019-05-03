import fire

from models import im_attn_ee_rnn_attn_dec_copy_net
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_attn_ee_rnn_attn_dec_copy_net
    fire.Fire(cls)
