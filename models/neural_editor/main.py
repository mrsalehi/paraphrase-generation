import sys
from pathlib import Path

from models import neural_editor
from models.common.config import Config


def main():
    config_path = sys.argv[1]
    data_dir = Path(sys.argv[2])
    try:
        mode = sys.argv[3]
    except:
        mode = 'train'

    config = Config.from_file(config_path)
    config.put('model_dir', str(data_dir / config.model_dir))
    if mode == 'train':
        neural_editor.train(config, data_dir)


if __name__ == '__main__':
    main()
