from tqdm import tqdm

from models.common import vocab
from models.im_aeradp_com_wa.input import input_fn_from_gen_multi
from models.im_attn_ee_rnn_attn_dec_copy_net.paraphrase_gen import create_formulas
from models.neural_editor.paraphrase_gen import read_plan, clean_sentence


def generate(estimator, plan_path, checkpoint_path, config, V):
    vocab.create_vocab_lookup_tables(V)

    batch_size = config.optim.batch_size
    if config.get('editor.use_beam_decoder', False):
        beam_width = config.editor.beam_width
    else:
        beam_width = 1

    plans = read_plan(plan_path)
    formulas, formula2plan = create_formulas(plans, config)

    num_edit_vectors = int(len(formulas) / len(plans))

    formula_gen = lambda: iter(formulas)
    output = estimator.predict(
        input_fn=lambda: input_fn_from_gen_multi(formula_gen, vocab.create_vocab_lookup_tables(V), batch_size),
        checkpoint_path=checkpoint_path
    )

    # plan2paraphrase = [[None for _ in range(num_edit_vectors)] for _ in range(len(plans))]
    plan2paraphrase = [[None for _ in evs] for b, evs in plans]
    plan2attn_weight = [[None for _ in evs] for b, evs in plans]
    plan2dec_base_attn_weight = [[None for _ in evs] for b, evs in plans]
    plan2dec_mev_attn_weight = [[None for _ in evs] for b, evs in plans]

    for i, o in enumerate(tqdm(output, total=len(formulas))):
        paraphrases = [clean_sentence(j.decode('utf8')) for j in o['joined']]
        assert len(paraphrases) == beam_width

        plan_index, edit_index = formula2plan[i]
        plan2paraphrase[plan_index][edit_index] = paraphrases

        if 'tmee_attentions_st_enc_self' in list(o.keys()):
            st = (o['tmee_attentions_st_enc_self'], o['tmee_attentions_st_dec_self'], o['tmee_attentions_st_dec_enc'])
            ts = (o['tmee_attentions_ts_enc_self'], o['tmee_attentions_ts_dec_self'], o['tmee_attentions_ts_dec_enc'])

            plan2attn_weight[plan_index][edit_index] = (st, ts)

        if 'dec_alg_base' in o:
            plan2dec_base_attn_weight[plan_index][edit_index] = o['dec_alg_base']

        if 'dec_alg_mev' in o:
            plan2dec_mev_attn_weight[plan_index][edit_index] = o['dec_alg_mev']

    assert len(plans) == len(plan2paraphrase)

    return plan2paraphrase, plan2attn_weight, plan2dec_base_attn_weight, plan2dec_mev_attn_weight
