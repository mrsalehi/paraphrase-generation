import fire

from models import im_vev_shared_enc
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_vev_shared_enc
    fire.Fire(cls)
