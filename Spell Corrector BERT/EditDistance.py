def edit_step(word):
    """
    All edits that are one edit away from `word`.
    """
    letters = (
        "ઁંઃઅઆઇઈઉઊઋઌઍએઐઑઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલળવશષસહ઼ઽાિીુૂૃૄૅેૈૉોૌ્ૐૠૡૢૣ૱"
    )
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(self, word):
    """
    All edits that are two edits away from `word`.
    """
    return (e2 for e1 in self.edit_step(word) for e2 in self.edit_step(e1))


def known(self, words):
    """
    The subset of `words` that appear in the dictionary of WORDS.
    """
    return set(w for w in words if w in self.WORDS)


def edit_candidates(self, word, assume_wrong=False, fast=False):
    """
    Generate possible spelling corrections for word.
    """

    if fast:
        ttt = self.known(self.edit_step(word)) or {word}
    else:
        ttp = self.known(self.edits2(word))
        ftp = self.known(self.edit_step(word))
        ttp.update(ftp)
        ttt = ttp or {word}

    ttt = self.known([word]) | ttt
    return list(ttt)
