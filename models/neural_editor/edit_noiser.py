import numpy as np


class EditNoiser:

    def __init__(self, ident_pr=0.1, attend_pr=0.5):
        self.ident_pr = ident_pr
        self.attend_pr = attend_pr

    def __call__(self, example):
        """Return a batch of noisy EditExamples.

        Does not modify the original EditExamples.
        """
        return self._noise(example)

    def dropout_split(self, word_list):
        if len(word_list) == 0:
            return []

        pr_list = self.attend_pr
        num_sampled = np.argmax(np.random.multinomial(1, pr_list)) + 1
        num_sampled = min(num_sampled, len(word_list))
        choice_index = np.random.choice(np.arange(len(word_list)), num_sampled, replace=False)

        mask = np.zeros(len(word_list), dtype=bool)
        mask[choice_index] = True

        warray = np.array(word_list)
        return warray[np.invert(mask)].tolist(), warray[mask].tolist()

    def _noise(self, ex):
        """Return a noisy EditExample.

        Note: this strategy is only appropriate for diff-style EditExamples.

        Args:
            ex (EditExample)

        Returns:
            EditExample: a new example. Does not modify the original example.
        """
        src_words, tgt_words, insert_words, delete_words = ex
        ident_map = np.random.binomial(1, self.ident_pr)
        if ident_map:
            return ex
        else:
            src_approx, removed_src_words = self.dropout_split(src_words)
            tgt_approx, removed_tgt_words = self.dropout_split(tgt_words)
            return (src_approx, tgt_approx, insert_words, delete_words)

    @classmethod
    def from_config(cls, config):
        if config.get('config.editor.enable_noiser', False):
            noiser = EditNoiser(config.editor.ident_pr, config.editor.attend_pr)
        else:
            noiser = None

        return noiser


if __name__ == '__main__':
    s = "How do I gain healthy weight without eating junk ?".split()
    t = "How do I gain weight healthily without getting fat ?".split()

    en = EditNoiser(0, [0.9, 0.1])
    ex = en((s, t, [], []))

    print(ex)
