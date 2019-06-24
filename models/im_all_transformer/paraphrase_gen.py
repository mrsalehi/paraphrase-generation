from typing import Dict, Sequence

import numpy as np
from bpemb import BPEmb
from tqdm import tqdm

from models.common import vocab, util, subtoken_encoder
from models.common.util import save_tsv
from models.im_all_transformer.input import parse_instance, map_word_to_sub_words, map_str_to_bytes, \
    input_fn_from_gen_multi, get_process_example_fn
from models.neural_editor import paraphrase_gen
from models.neural_editor.edit_noiser import EditNoiser
from models.neural_editor.paraphrase_gen import read_plan


def create_formulas(plans, config):
    noiser = EditNoiser.from_config(config)
    free_set = util.get_free_words_set() if config.get('editor.use_free_set', False) else None

    process_instance_fn = get_process_example_fn(config)

    formulas = []
    formula2plan = []
    for i, (base, edits) in enumerate(plans):
        output = ''
        basic_formula = (base, output)
        for j, edit_vector_pair in enumerate(edits):
            instance = basic_formula + edit_vector_pair
            instance = parse_instance('\t'.join(instance), noiser, free_set)
            formula = process_instance_fn(instance)
            formulas.append(formula)
            formula2plan.append((i, j))

    return formulas, formula2plan


def clean_sentence(sent):
    for tok in vocab.SPECIAL_TOKENS:
        sent = sent.replace(tok, '')

    return sent.strip()


def get_t2t_str_tokens_fn(config):
    encoder = vocab.get_t2t_subword_encoder_instance(config)

    def converter(word_ids):
        try:
            index = list(word_ids).index(subtoken_encoder.PAD_ID)
            words = encoder.decode_list(word_ids[:index])
        except ValueError:  # No PAD found in sequence
            words = encoder.decode_list(word_ids)

        return words

    return converter


def get_bpemb_str_tokens_fn(config):
    bpemb = vocab.get_bpemb_instance(config)

    def converter(word_ids):
        try:
            index = list(word_ids).index(vocab.SPECIAL_TOKENS.index(vocab.PAD_TOKEN))
            word_ids = word_ids - 1
            words = [bpemb.pieces[i] for i in word_ids[:index]]
        except ValueError:  # No PAD found in sequence
            word_ids = word_ids - 1
            words = [bpemb.pieces[i] for i in word_ids]

        return words

    return converter


def clean_sub_word_sentence(word_ids: np.array, bpemb: BPEmb):
    # Extra padding token is remove in BPEmb
    word_ids = word_ids - 1
    try:
        index = list(word_ids).index(bpemb.EOS)
        words = bpemb.decode_ids(word_ids[:index])
    except ValueError:  # No EOS found in sequence
        words = bpemb.decode_ids(word_ids)

    return words


def get_bpemb_decode_fn(config):
    bpemb = vocab.get_bpemb_instance(config)

    def decode(o: Dict) -> Sequence[str]:
        paraphrases = [clean_sentence(j.decode('utf8')).split() for j in o['joined']]
        paraphrases = bpemb.decode(paraphrases)
        return paraphrases

    return decode


def get_single_word_decode_fn(config):
    def decode(o: Dict) -> Sequence[str]:
        paraphrases = [clean_sentence(j.decode('utf8')) for j in o['joined']]
        return paraphrases

    return decode


def get_t2t_subword_decode_fn(config):
    encoder = vocab.get_t2t_subword_encoder_instance(config)

    def decode_sent(word_ids):
        word_ids = list(word_ids)
        try:
            index = word_ids.index(subtoken_encoder.EOS_ID)
            words = encoder.decode(word_ids[:index], disable_tokenizer=True)
        except ValueError:  # No EOS found in sequence
            # try:
            #     index = word_ids.index(subtoken_encoder.UNK_ID)
            #     words = encoder.decode(word_ids[:index], disable_tokenizer=True)
            # except ValueError:
            words = encoder.decode(word_ids, disable_tokenizer=True)

        return words

    def decode(o: Dict) -> Sequence[str]:
        paraphrases = [' '.join(decode_sent(j)) for j in o['decoded_ids']]
        return paraphrases

    return decode


def get_convert_o_to_paraphrases_fn(config):
    decode_fn = get_single_word_decode_fn(config)
    if config.editor.get('use_sub_words', False):
        if config.editor.get('use_t2t_sub_words', False):
            decode_fn = get_t2t_subword_decode_fn(config)
        else:
            decode_fn = get_bpemb_decode_fn(config)

    convertor_fn = lambda o: decode_fn(o)

    return convertor_fn


def get_str_token_converter(config):
    if config.editor.get('use_sub_words', False):
        if config.editor.get('use_t2t_sub_words', False):
            fn = get_t2t_str_tokens_fn(config)
        else:
            fn = get_bpemb_str_tokens_fn(config)
    else:
        return None

    convertor_fn = lambda o: fn(o)

    return convertor_fn


def generate(estimator, plan_path, checkpoint_path, config, V):
    vocab.create_vocab_lookup_tables(V)

    batch_size = config.optim.batch_size
    beam_width = config.editor.decoder.beam_size

    plans = read_plan(plan_path)
    formulas, formula2plan = create_formulas(plans, config)

    formula_gen = lambda: iter(formulas)
    output = estimator.predict(
        input_fn=lambda: input_fn_from_gen_multi(formula_gen, vocab.create_vocab_lookup_tables(V), batch_size),
        checkpoint_path=checkpoint_path
    )

    plan2paraphrase = [[None for _ in evs] for b, evs in plans]
    plan2pq = [[None for _ in evs] for b, evs in plans]
    plan2attn_weight = [[None for _ in evs] for b, evs in plans]

    get_paraphrases = get_convert_o_to_paraphrases_fn(config)
    get_str_tokens = get_str_token_converter(config)

    save_attentions = config.get("eval.save_attentions", False)

    for i, o in enumerate(tqdm(output, total=len(formulas))):
        paraphrases = get_paraphrases(o)
        assert len(paraphrases) == beam_width

        plan_index, edit_index = formula2plan[i]
        plan2paraphrase[plan_index][edit_index] = paraphrases

        if not save_attentions:
            continue

        if 'tmee_attentions_st_enc_self' in list(o.keys()):
            st = (o['tmee_attentions_st_enc_self'], o['tmee_attentions_st_dec_self'], o['tmee_attentions_st_dec_enc'])
            ts = (o['tmee_attentions_ts_enc_self'], o['tmee_attentions_ts_dec_self'], o['tmee_attentions_ts_dec_enc'])

            plan2attn_weight[plan_index][edit_index] = (st, ts)

            plan2pq[plan_index][edit_index] = (
                ' '.join(get_str_tokens(o['src_words'])),
                ' '.join(get_str_tokens(o['tgt_words']))
            )

    assert len(plans) == len(plan2paraphrase)

    return plan2paraphrase, plan2attn_weight, plan2pq


def save_outputs(outputs, output_path):
    paras, attn_weights, pqs = outputs

    paraphrase_gen.save_attn_weights(attn_weights, '%s.attn_weights' % output_path)

    para_flatten = paraphrase_gen.flatten(paras)
    save_tsv(output_path, para_flatten)

    pq_flatten = paraphrase_gen.flatten(pqs)
    save_tsv('%s.pqs' % output_path, pq_flatten)
