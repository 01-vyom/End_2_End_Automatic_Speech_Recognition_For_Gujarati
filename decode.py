# Importing libraries
import pandas as pd
import pickle
import dill
import warnings
import re
import tensorflow as tf
import numpy as np
import codecs
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")
dill._dill._reverse_typemap["ClassType"] = type


lmword_data = None


def lm(string, total=2501841):
    """
    4 Gram Word-Level Langauage Model which returns the probability based on the input words.
    Args:
      string (string): Words separated by space.
      total (int): Total words in the corpus usd to form the model.
    Returns:
      float: Probability.
    """

    sl = list(string.split(" "))
    total = lmword_data[4]
    ul = []
    bl = []
    tl = []
    ql = []
    up = []
    bp = []
    tp = []
    qp = []
    final = 1
    for i in sl:
        try:
            ul.append(lmword_data[0][i])
        except:
            ul.append(0)

    for i in range(0, len(sl) - 1):
        s = [sl[i], sl[i + 1]]
        stri = " ".join([str(elem) for elem in s])
        count = lmword_data[1][stri]
        bl.append(count)

    for i in range(0, len(sl) - 2):
        s = [sl[i], sl[i + 1], sl[i + 2]]
        stri = " ".join([str(elem) for elem in s])
        count = lmword_data[2][stri]
        tl.append(count)

    for i in range(0, len(sl) - 3):
        s = [sl[i], sl[i + 1], sl[i + 2], sl[i + 3]]
        stri = " ".join([str(elem) for elem in s])
        count = lmword_data[3][stri]
        ql.append(count)

    for v in range(0, len(ul)):
        up.append(ul[v] / total)
        if len(sl) == 1:
            return ul[v] / total + 1 / total

    for v in range(0, len(bl)):
        try:
            k = bl[v] / ul[v]
        except:
            k = 0.01
        bp.append(0.2 * up[v + 1] + 0.8 * k)
        if len(sl) == 2:
            return 0.2 * up[v + 1] + 0.8 * k + 1 / total

    for v in range(0, len(tl)):
        try:
            k = tl[v] / bl[v]
        except:
            k = 0.01
        tp.append(0.1 * up[v + 2] + 0.3 * bp[v + 1] + 0.6 * k)
        if len(sl) == 3:
            return 0.1 * up[v + 2] + 0.3 * bp[v + 1] + 0.6 * k + 1 / total

    for v in range(0, len(ql)):
        try:
            k = ql[v] / tl[v]
        except:
            k = 0.01
        qp.append(0.02 * up[v + 3] + 0.08 * bp[v + 2] + 0.3 * tp[v + 1] + 0.6 * k)

    for i in qp:
        final *= i

    return final + 1 / total


class LanguageModel:
    ##BiGram Character Level Language Model

    "simple language model: word list for token passing, char bigrams for beam search"

    def __init__(self, fn, classes):
        "read text from file to generate language model"
        self.initWordList(fn)
        self.initCharBigrams(fn, classes)

    def initWordList(self, fn):
        "internal init of word list"
        txt = open(fn).read().lower()
        words = re.findall(r"\w+", txt)
        self.words = list(filter(lambda x: x.isalpha(), words))

    def initCharBigrams(self, fn, classes):
        "internal init of character bigrams"

        # init bigrams with 0 values
        self.bigram = {c: {d: 0 for d in classes} for c in classes}

        # go through text and add each char bigram
        txt = codecs.open(fn, "r", "utf8").read()
        for i in range(len(txt) - 1):
            first = txt[i]
            second = txt[i + 1]

            # ignore unknown chars
            if first not in self.bigram or second not in self.bigram[first]:
                continue

            self.bigram[first][second] += 1

    def getCharBigram(self, first, second):
        "probability of seeing character 'first' next to 'second'"
        first = first if first else " "  # map start to word beginning
        second = second if second else " "  # map end to word end

        # number of bigrams starting with given char
        numBigrams = sum(self.bigram[first].values())
        if numBigrams == 0:
            return 0
        return self.bigram[first][second] / numBigrams

    def getWordList(self):
        "get list of unique words"
        return self.words


def prefix_beam_search(
    ctc,
    alphabet,
    lm=None,
    lmchar=None,
    k=25,
    alphac=0.8,
    alphaw=0.3,
    beta=5,
    prune=0.001,
):
    """
    Performs prefix beam search on the output of a CTC network.
    Args:

      ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
      alphabet (list of strings): The alphabets of the Gujarati language + [space_token, end_token, blank_token]
      lm (func): Word Level Language model function. Should take as input a string and output a probability.
      lmchar (object): Character-Level Language model Object. Should take 2 characters ass input anf give probability of both of them occuring together.
      k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
      alphac (float): The character level language model weight. Should usually be between 0 and 1.
      alphaw (float): The word level language model weight. Should usually be between 0 and 1.
      beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
      prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.
    Returns:
      string: The decoded CTC output.
    """

    lm = (
        (lambda l: 1) if lm is None else lm
    )  # if no LM is provided, just set to function returning 1
    W = lambda l: re.findall(
        r"[\u0A81\u0A82\u0A83\u0A85\u0A86\u0A87\u0A88\u0A89\u0A8A\u0A8B\u0A8C\u0A8D\u0A8F\u0A90\u0A91\u0A93\u0A94\u0A95\u0A96\u0A97\u0A98\u0A99\u0A9A\u0A9B\u0A9C\u0A9D\u0A9E\u0A9F\u0AA0\u0AA1\u0AA2\u0AA3\u0AA4\u0AA5\u0AA6\u0AA7\u0AA8\u0AAA\u0AAB\u0AAC\u0AAD\u0AAE\u0AAF\u0AB0\u0AB2\u0AB3\u0AB5\u0AB6\u0AB7\u0AB8\u0AB9\u0ABC\u0ABD\u0ABE\u0ABF\u0AC0\u0AC1\u0AC2\u0AC3\u0AC4\u0AC5\u0AC7\u0AC8\u0AC9\u0ACB\u0ACC\u0ACD\u0AD0\u0AE0\u0AE1\u0AE2\u0AE3\u0AF1]+[\s|>]",
        l,
    )
    F = ctc.shape[1]
    ctc = np.vstack(
        (np.zeros(F), ctc)
    )  # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ""
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:

            if len(l) > 0 and l[-1] == ">":
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2gujdata.txt

                # STEP 3: “Extending” with a blank
                if c == "%":
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3

                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c

                    init = " "
                    if len(l):
                        init = l[-1]
                    lmchar_prob = 1
                    if lmchar:
                        lmchar_prob = lmchar.getCharBigram(init, c) ** alphac

                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l] * lmchar_prob
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                    # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(" ", "")) > 0 and c in (" ", ">"):
                        lm_prob = lm(l_plus.strip(" >")) ** alphaw
                        Pnb[t][l_plus] += (
                            lm_prob
                            * ctc[t][c_ix]
                            * (Pb[t - 1][l] + Pnb[t - 1][l])
                            * lmchar_prob
                        )
                    else:
                        Pnb[t][l_plus] += (
                            ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l]) * lmchar_prob
                        )
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += (
                            ctc[t][-1]
                            * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                            * lmchar_prob
                        )
                        Pnb[t][l_plus] += (
                            ctc[t][c_ix] * Pnb[t - 1][l_plus] * lmchar_prob
                        )
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7
    return A_prev[0].strip(">")


def init(alphabet):
    """
    Initializes Word Level LM Data an Character level LM Object.
    Args:
      alphabet (list of strings): The alphabets of the Gujarati language + [space_token, end_token, blank_token]
    Returns:
      list: list of dictionaires required for Word Level language model
      lmchar: Character Level Language Model Object
    """

    ##Character Level Language Model Init
    lmdata = "Data Files/gujdata.txt"
    classes = "".join(alphabet[:-1])
    lmchar = LanguageModel(lmdata, classes)

    ##Word Level Language Model Init
    with open(lmdata, "r") as file:
        corpus = file.read().replace("\n", "")
    cor_ls = corpus.split(" ")
    le = len(cor_ls)
    temp_ls1 = []
    for i in range(le):
        temp_ls1.append(str(cor_ls[i]))

    temp_ls2 = []
    for i in range(le - 1):
        temp_ls2.append(cor_ls[i] + " " + cor_ls[i + 1])

    temp_ls3 = []
    for i in range(le - 2):
        temp_ls3.append(cor_ls[i] + " " + cor_ls[i + 1] + " " + cor_ls[i + 2])

    temp_ls4 = []
    for i in range(le - 3):
        temp_ls4.append(
            cor_ls[i] + " " + cor_ls[i + 1] + " " + cor_ls[i + 2] + " " + cor_ls[i + 3]
        )

    uni_dct = Counter(temp_ls1)
    bi_dct = Counter(temp_ls2)
    tri_dct = Counter(temp_ls3)
    quad_dct = Counter(temp_ls4)

    return [uni_dct, bi_dct, tri_dct, quad_dct, le], lmchar


def decode(lmchar, alphabet, model_name):
    """
    Performs all types of decoding on the output of a CTC network and saves them in one .CSV file.
    Args:
      alphabet (list of strings): The alphabets of the Gujarati language + [space_token, end_token, blank_token]
      lmchar (object): Character-Level Language model Object. Should take 2 characters ass input anf give probability of both of them occuring together.
      model_name (string): Name of the model for which the the hypothesis will be decoded and saved.
    """
    # call the decoder_and_merger given output from trained model
    refs = dill.load(open("refs_" + model_name + ".pickle", "rb"))
    hyps = dill.load(open("hyps_" + model_name + ".pickle", "rb"))
    hyps_beam = []  # Prefix Decoding with No LM
    hyps_greedy = []  # Greedy Decoding
    hyps_beam_lm = []  # Prefix Decoding with Both LMs
    hyps_beam_wlm = []  # Prefix Decoding with WLM
    hyps_beam_clm = []  # Prefix Decoding with CLM

    for i in range(len(hyps)):
        # Prefix Beam Decoding with WLM.
        prefix = prefix_beam_search(
            hyps[i],
            alphabet,
            lm=lm,
            lmchar=None,
            k=45,
            alphac=0.0,
            alphaw=0.03,
            beta=3,
            prune=0.001,
        )
        hyps_beam_wlm.append(prefix)
        # Prefix Beam Decoding with CLM.
        prefix = prefix_beam_search(
            hyps[i],
            alphabet,
            lm=None,
            lmchar=lmchar,
            k=45,
            alphac=0.4,
            alphaw=0.0,
            beta=3,
            prune=0.001,
        )
        hyps_beam_clm.append(prefix)
        # Prefix Beam Decoding with both LM.
        prefix = prefix_beam_search(
            hyps[i],
            alphabet,
            lm=lm,
            lmchar=lmchar,
            k=45,
            alphac=0.4,
            alphaw=0.03,
            beta=3,
            prune=0.001,
        )
        hyps_beam_lm.append(prefix)
        # Prefix Beam Decoding without LMs.
        prefix = prefix_beam_search(
            hyps[i],
            alphabet,
            lm=None,
            lmchar=None,
            k=45,
            alphac=0.0,
            alphaw=0.0,
            beta=0,
            prune=0.001,
        )
        hyps_beam.append(prefix)
        # Greedy Decoding
        output_text = ""
        for timestep in hyps[i]:
            output_text += alphabet[tf.math.argmax(timestep)]
        final = output_text[0]
        for k in range(1, len(output_text)):
            if output_text[k] != output_text[k - 1]:
                if (
                    output_text[k] != ">"
                    and output_text[k] != "%"
                    and output_text[k] != "#"
                ):
                    final += output_text[k]
        final = final.replace("%", "")
        hyps_greedy.append(final)

    # Saving all the hypothesis and references according to model name and decoding type
    Scripts = pd.DataFrame(
        [re.sub(r"[\s]+", " ", item.strip()) for item in hyps_greedy],
        columns=["Hypothesis_Greedy_" + model_name],
    )
    Scripts["Hypothesis_Prefix_" + model_name] = [
        re.sub(r"[\s]+", " ", item.strip()) for item in hyps_beam
    ]
    Scripts["Hypothesis_Prefix_LM_" + model_name] = [
        re.sub(r"[\s]+", " ", item.strip()) for item in hyps_beam_lm
    ]
    Scripts["Hypothesis_Prefix_CLM_" + model_name] = [
        re.sub(r"[\s]+", " ", item.strip()) for item in hyps_beam_clm
    ]
    Scripts["Hypothesis_Prefix_WLM_" + model_name] = [
        re.sub(r"[\s]+", " ", item.strip()) for item in hyps_beam_wlm
    ]
    Scripts["Actual_" + model_name] = refs
    Scripts.to_csv(model_name + "_ALL_DECODING.csv", index=False)


if __name__ == "__main__":
    space_token = " "
    end_token = ">"
    blank_token = "%"
    gujarati_alphabet = [
        "\u0A81",
        "\u0A82",
        "\u0A83",
        "\u0A85",
        "\u0A86",
        "\u0A87",
        "\u0A88",
        "\u0A89",
        "\u0A8A",
        "\u0A8B",
        "\u0A8C",
        "\u0A8D",
        "\u0A8F",
        "\u0A90",
        "\u0A91",
        "\u0A93",
        "\u0A94",
        "\u0A95",
        "\u0A96",
        "\u0A97",
        "\u0A98",
        "\u0A99",
        "\u0A9A",
        "\u0A9B",
        "\u0A9C",
        "\u0A9D",
        "\u0A9E",
        "\u0A9F",
        "\u0AA0",
        "\u0AA1",
        "\u0AA2",
        "\u0AA3",
        "\u0AA4",
        "\u0AA5",
        "\u0AA6",
        "\u0AA7",
        "\u0AA8",
        "\u0AAA",
        "\u0AAB",
        "\u0AAC",
        "\u0AAD",
        "\u0AAE",
        "\u0AAF",
        "\u0AB0",
        "\u0AB2",
        "\u0AB3",
        "\u0AB5",
        "\u0AB6",
        "\u0AB7",
        "\u0AB8",
        "\u0AB9",
        "\u0ABC",
        "\u0ABD",
        "\u0ABE",
        "\u0ABF",
        "\u0AC0",
        "\u0AC1",
        "\u0AC2",
        "\u0AC3",
        "\u0AC4",
        "\u0AC5",
        "\u0AC7",
        "\u0AC8",
        "\u0AC9",
        "\u0ACB",
        "\u0ACC",
        "\u0ACD",
        "\u0AD0",
        "\u0AE0",
        "\u0AE1",
        "\u0AE2",
        "\u0AE3",
        "\u0AF1",
    ]
    alphabet = gujarati_alphabet + [space_token, end_token, blank_token]
    lmword_data, lmchar = init(alphabet)
    # Model Specific Reference and Hypothesis.
    model_name = "temp_model"
    decode(lmchar, alphabet, model_name)