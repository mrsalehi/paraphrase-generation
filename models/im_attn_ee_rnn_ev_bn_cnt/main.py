import fire

from models import im_attn_ee_rnn_ev_bn_cnt
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_attn_ee_rnn_ev_bn_cnt
    fire.Fire(cls)
