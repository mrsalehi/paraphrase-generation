import tensorflow as tf

import models.im_all_transformer.edit_encoder as base_edit_encoder
from models.common.config import Config
from models.im_all_transformer.encoder import TransformerEncoder

OPS_NAME = 'edit_encoder'


class TransformerMicroEditExtractor(base_edit_encoder.TransformerMicroEditExtractor):
    def __init__(self, embedding_layer, mev_projection: tf.layers.Dense,
                 sent_encoder: TransformerEncoder, params, **kwargs):
        super().__init__(embedding_layer, mev_projection, params, **kwargs)

        self.target_encoder = sent_encoder.encoder
        self.target_encoder.set_save_alignment_history(params.save_attentions)


class EditEncoder(base_edit_encoder.EditEncoder):
    def __init__(self, config, sent_encoder, **kwargs):
        super().__init__(config, **kwargs)

        extractor_config = Config.merge_to_new([config.editor.transformer, config.editor.edit_encoder.extractor])
        extractor_config.put('save_attentions', config.get('eval.save_attentions', False))

        self.mev_extractor = TransformerMicroEditExtractor(
            self.embedding_layer,
            self.micro_ev_projection,
            sent_encoder,
            extractor_config
        )
