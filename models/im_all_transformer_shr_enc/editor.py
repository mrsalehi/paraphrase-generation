import models.im_all_transformer.editor as base_editor
from models.common.config import Config
from models.im_all_transformer_shr_enc.edit_encoder import EditEncoder


class Editor(base_editor.Editor):
    def __init__(self, config: Config):
        super().__init__(config)

        self.edit_encoder = EditEncoder(config, self.encoder)
