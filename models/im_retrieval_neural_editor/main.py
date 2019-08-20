import fire

from models import im_retrieval_neural_editor
from models.neural_editor.main import ModelRunner

if __name__ == '__main__':
    cls = ModelRunner
    cls.model = im_retrieval_neural_editor
    fire.Fire(cls)
