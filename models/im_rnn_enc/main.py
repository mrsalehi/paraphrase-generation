import sys
from pathlib import Path

from models import im_rnn_enc
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

    config = Config.from_file(config_path)
    config.put('model_dir', str(data_dir / config.model_dir))

    put_epoch_num(config, data_dir)

    print(config_path)
    print(config)

    if mode == 'train':
        im_rnn_enc.train(config, data_dir)


if __name__ == '__main__':
    main()
