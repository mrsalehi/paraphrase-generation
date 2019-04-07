import fire

from models import im_cnct_agn
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_cnct_agn
    fire.Fire(cls)
