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
        pr_list = [1.0 - self.attend_pr, self.attend_pr]
        if len(word_list) > 0:
            num_sampled = np.random.choice(np.arange(len(pr_list)), 1, p=pr_list)
            num_sampled = min(num_sampled, len(word_list))
            choice_index = np.random.choice(np.arange(len(word_list)), num_sampled, replace=False)
            mask = np.zeros(len(word_list), dtype=bool)
            mask[choice_index] = True
            warray = np.array(word_list)
            return (warray[mask]).tolist(), (warray[np.invert(mask)]).tolist()
        else:
            return [], []

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
            return (src_words, [], [], tgt_words)
        else:
            insert_exact, insert_approx = self.dropout_split(insert_words)
            delete_exact, delete_approx = self.dropout_split(delete_words)
            return (src_words, tgt_words, insert_approx, delete_approx)

    @classmethod
    def from_config(cls, config):
        if config.get('config.editor.enable_noiser', False):
            noiser = EditNoiser(config.editor.ident_pr, config.editor.attend_pr)
        else:
            noiser = None

        return noiser
