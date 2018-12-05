import sys
from pathlib import Path

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

    if mode == 'predict':
        # neural_editor.predict_cmd(config, data_dir, checkpoint)
        neural_editor.augment_meta_test(config,
                                        '/Users/amirhosein/Development/PycharmProjects/dbpedia/pdata/meta_test/5w1s_e100.pkl',
                                        data_dir, checkpoint)


if __name__ == '__main__':
    main()
