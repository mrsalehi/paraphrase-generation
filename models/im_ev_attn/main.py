import fire

from models import im_ev_attn
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_ev_attn
    fire.Fire(cls)
