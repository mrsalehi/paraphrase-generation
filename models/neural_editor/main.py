import sys
from pathlib import Path

import fire

from models import neural_editor
from models.common import util
from models.common.config import Config


def put_epoch_num(config, data_dir):
    p = data_dir / config.dataset.path / 'train.tsv'
    total_num_examples = util.get_num_total_lines(p)
    num_batch_per_epoch = total_num_examples // config.optim.batch_size
    num_epoch = config.optim.max_iters // num_batch_per_epoch + 1
    config.put('optim.num_epoch', num_epoch)


def main():
    config_path = sys.argv[1]
    data_dir = Path(sys.argv[2])
    try:
        mode = sys.argv[3]
    except:
        mode = 'train'

    try:
        checkpoint = sys.argv[4]
    except:
        checkpoint = None

    config = Config.from_file(config_path)
    config.put('model_dir', str(data_dir / config.model_dir))

    put_epoch_num(config, data_dir)

    print(config_path)
    print(config)

    if mode == 'train':
        neural_editor.train(config, data_dir)

    if mode == 'eval':
        neural_editor.eval(config, data_dir, checkpoint)

    if mode == 'augment':
        # neural_editor.predict_cmd(config, data_dir, checkpoint)
        meta_test = sys.argv[5]
        neural_editor.augment_meta_test(config,
                                        meta_test,
                                        data_dir, checkpoint)

    if mode == 'debug':
        try:
            checkpoint = sys.argv[4]
        except:
            checkpoint = None

        debug_dataset = sys.argv[5]

        neural_editor.augment_debug(config, debug_dataset, data_dir, checkpoint)


class ModelRunner(object):
    model = neural_editor

    def __init__(self, config, data_dir, checkpoint=None):
        self._data_dir = Path(data_dir)
        self._checkpoint = checkpoint

        self._config = Config.from_file(config)
        self._config.put('model_dir', str(self._data_dir / self._config.model_dir))
        self._put_epoch_num()

        print("Model:", self.model.NAME)
        print("Config:")
        print(self._config)
        print()

    def _put_epoch_num(self):
        p = self._data_dir / self._config.dataset.path / 'train.tsv'
        total_num_examples = util.get_num_total_lines(p)
        num_batch_per_epoch = total_num_examples // self._config.optim.batch_size
        num_epoch = self._config.optim.max_iters // num_batch_per_epoch + 1

        self._config.put('optim.num_epoch', num_epoch)

    def train(self):
        self.model.train(self._config, self._data_dir)

    def eval(self):
        self.model.eval(self._config, self._data_dir, self._checkpoint)

    def augment(self, meta_test_path):
        self.model.augment_meta_test(self._config, meta_test_path, self._data_dir, self._checkpoint)

    def debug(self, debug_dataset):
        self.model.augment_debug(self._config, debug_dataset, self._data_dir, self._checkpoint)


if __name__ == '__main__':
    fire.Fire(ModelRunner)
