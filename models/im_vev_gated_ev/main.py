import fire

from models import im_vev_gated_ev
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_vev_gated_ev
    fire.Fire(cls)
