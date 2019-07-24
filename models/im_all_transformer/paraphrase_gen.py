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


def get_attn_pairs(attn_map, layer, head, query_len, key_len, add_second_target=False):
    p2q_attn = attn_map[layer][head][:query_len, :key_len]

    # attn_tgts = np.argmax(p2q_attn, axis=1)
    # attn_srcs = np.arange(query_len)
    #
    # pairs = np.concatenate([
    #     attn_srcs.reshape((-1, 1)),
    #     attn_tgts.reshape((-1, 1))
    # ], axis=1)

    attn_tgts = p2q_attn.argsort(axis=1)[:, -2:][:, ::-1]
    attn_srcs = np.arange(query_len)

    first_targets = attn_tgts[:, 0]
    pairs = np.concatenate([
        attn_srcs.reshape((-1, 1)),
        first_targets.reshape((-1, 1))
    ], axis=1)

    if add_second_target:
        probs = p2q_attn[attn_srcs.reshape((-1, 1)), attn_tgts]
        probs = util.softmax(probs, axis=1)

        valid_second_targets = probs[:, 1] > 0.43
        second_srcs = attn_srcs[valid_second_targets]
        second_tgts = attn_tgts[valid_second_targets, 1]

        pairs_from_second_targets = np.concatenate([
            second_srcs.reshape((-1, 1)),
            second_tgts.reshape((-1, 1))
        ], axis=1)

        pairs = np.concatenate([pairs, pairs_from_second_targets], axis=0)

    return pairs


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
    plan2mevs = [[None for _ in evs] for b, evs in plans]

    get_paraphrases = get_convert_o_to_paraphrases_fn(config)
    get_str_tokens = get_str_token_converter(config)

    save_attentions = config.get("eval.save_attentions", False)
    save_mev = config.get('eval.save_mev', False)
    mev_layer = config.get('eval.save_mev_layer', 0)
    mev_head = config.get('eval.save_mev_head', None)
    mev_add_cls_tok = config.get('eval.save_mev_cls_tok', False)
    mev_add_second_tgt = config.get('eval.save_mev_add_second_tgt', False)

    for i, o in enumerate(tqdm(output, total=len(formulas))):
        paraphrases = get_paraphrases(o)
        assert len(paraphrases) == beam_width

        plan_index, edit_index = formula2plan[i]
        plan2paraphrase[plan_index][edit_index] = paraphrases

        if 'tmee_attentions_st_enc_self' not in list(o.keys()) or \
                'src_words' not in list(o.keys()) or \
                'tgt_words' not in list(o.keys()):
            continue

        src = get_str_tokens(o['src_words'])
        tgt = get_str_tokens(o['tgt_words'])

        plan2pq[plan_index][edit_index] = (
            ' '.join(src),
            ' '.join(tgt)
        )

        st = (o['tmee_attentions_st_enc_self'], o['tmee_attentions_st_dec_self'], o['tmee_attentions_st_dec_enc'])
        ts = (o['tmee_attentions_ts_enc_self'], o['tmee_attentions_ts_dec_self'], o['tmee_attentions_ts_dec_enc'])

        if save_attentions:
            plan2attn_weight[plan_index][edit_index] = (st, ts)

        if save_mev:
            st_attn_map = st[-1]
            ts_attn_map = ts[-1]

            save_all_heads = mev_head is None
            num_heads = st_attn_map.shape[1]

            heads = range(num_heads) if save_all_heads else [mev_head]

            st_attn_pairs = np.ones(shape=[0, 2], dtype=np.uint8)
            ts_attn_pairs = np.ones(shape=[0, 2], dtype=np.uint8)

            for i, h in enumerate(heads):
                query_len = (len(src) + 1) if mev_add_cls_tok else len(src)
                st_pairs = get_attn_pairs(st_attn_map, mev_layer, h, query_len, len(tgt), mev_add_second_tgt)
                st_attn_pairs = np.concatenate([st_attn_pairs, st_pairs], axis=0)

                query_len = (len(tgt) + 1) if mev_add_cls_tok else len(tgt)
                ts_pairs = get_attn_pairs(ts_attn_map, mev_layer, h, query_len, len(src), mev_add_second_tgt)
                ts_attn_pairs = np.concatenate([ts_attn_pairs, ts_pairs], axis=0)

            plan2mevs[plan_index][edit_index] = (st_attn_pairs.astype(np.uint8), ts_attn_pairs.astype(np.uint8))

    assert len(plans) == len(plan2paraphrase)

    return plan2paraphrase, plan2attn_weight, plan2pq, plan2mevs


def save_outputs(outputs, output_path):
    paras, attn_weights, pqs, mevs = outputs

    paraphrase_gen.save_attn_weights(attn_weights, '%s.attn_weights' % output_path)
    paraphrase_gen.save_attn_weights(mevs, '%s.mevs' % output_path)

    para_flatten = paraphrase_gen.flatten(paras)
    save_tsv(output_path, para_flatten)

    pq_flatten = paraphrase_gen.flatten(pqs)
    save_tsv('%s.pqs' % output_path, pq_flatten)
