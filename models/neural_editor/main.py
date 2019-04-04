import os
from pathlib import Path

import fire

from models import neural_editor
from models.common import util
from models.common.config import Config


class ModelRunner(object):
    model = neural_editor

    def __init__(self, config, data_dir, checkpoint=None):
        self._data_dir = Path(data_dir)
        self._checkpoint = checkpoint

        self._config = Config.from_file(config)
        self._path_model_dir()
        self._put_epoch_num()

        print("Model:", self.model.NAME)
        print("Config:")
        print(self._config)
        print()

    def _put_epoch_num(self):
        p = self._data_dir / self._config.dataset.path / 'train.tsv'
        if not p.exists():
            return

        total_num_examples = util.get_num_total_lines(p)
        num_batch_per_epoch = total_num_examples // self._config.optim.batch_size
        num_epoch = self._config.optim.max_iters // num_batch_per_epoch + 1

        self._config.put('optim.num_epoch', num_epoch)

    def _path_model_dir(self):
        data_dir = self._config.get('gcloud_storage_data_dir', self._data_dir)
        self._config.put('model_dir', os.path.join(str(data_dir), self._config.model_dir))

    def train(self):
        self.model.train(self._config, self._data_dir)

    def eval(self):
        self.model.eval(self._config, self._data_dir, self._checkpoint)

    def augment(self, meta_test_path):
        self.model.augment_meta_test(self._config, meta_test_path, self._data_dir, self._checkpoint)

    def debug(self, debug_dataset):
        self.model.augment_debug(self._config, debug_dataset, self._data_dir, self._checkpoint)

    def gen_paraphrase(self, plan, output, batch_size=None, beam_width=None):
        self.model.generate_paraphrase(
            self._config,
            self._data_dir,
            self._checkpoint,
            plan, output,
            beam_width, batch_size
        )


if __name__ == '__main__':
    fire.Fire(ModelRunner)
