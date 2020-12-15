import difflib
import pandas as pd
import re
import numpy as np


def word_level(model_name, type):

    # CSV containing Hypothesis and ground truth sentences.
    Sentdata = pd.read_csv("PATH/TO/FILE/" + model_name + "_ALL_DECODING.csv")

    # Checking Data
    # print(Sentdata.head())

    # Ground Truth sentences.
    ground = list(Sentdata["Actual_" + model_name].values)
    # Hypothesis sentences for a particular decoding technique.
    pred = list(Sentdata["Hypothesis_" + type + model_name].values)

    list_err = []
    corr = []
    corwrong = []
    total = []
    tempdf1 = []
    tempdf2 = []
    actual_words = []
    predicted_words = []
    bool_same = []

    for i in range(len(ground)):
        final = pred[i].replace(".", "")
        y = ground[i].replace(".", "")
        asd = ""
        for i, s in enumerate(difflib.ndiff(final, y)):

            if s[2] == " " and s[0] == "+":
                asd += " "
            elif s[2] == " " and s[0] == "-":
                continue
            elif s[0] == "+" and s[2] != " ":
                continue
            elif s[0] == "-" and s[2] != " ":
                asd += s[2]
            elif s[0] == " ":
                asd += s[2]
        final = asd
        l1 = list(y.split(" "))
        l2 = list(final.split(" "))
        for j in range(len(l1)):
            if l1[j] != "":
                if l1[j] != l2[j]:
                    list_err.append(l1[j])
                    bool_same.append(0)
                else:
                    corr.append(l1[j])
                    bool_same.append(1)
                actual_words.append(l1[j])
                predicted_words.append(l2[j])
                total.append(l1[j])
            else:
                pass
    len(list_err)
    wrong = list_err.copy()

    # Saving Actual and Predicted word for a particular decoding technique to use later in character level analysis.
    pd.DataFrame(
        {
            "Actual_word": actual_words,
            "Predicted_word": predicted_words,
            "same?": bool_same,
        }
    ).to_csv("Actual_Predicted_" + type + model_name + "_ALLWORDS.csv", index=False)

    # Checking Data
    # print("wrong words count",len(wrong))
    # print("correct words count",len(corr))
    # print("total words count",len(total))
    # print("total words count",len(wrong)+len(corr))

    # Separating always correct, always incorrect and partially correct as well as incorrect.
    aux = list(set(corr).intersection(set(wrong)))
    fwrong = [[x, wrong.count(x)] for x in set(wrong)]
    temp = []
    for e in fwrong:
        if e[0] in aux:
            f = [e[0]] * e[1]
            corwrong.extend(f)
            tempdf1.append([e[0], e[1]])
        else:
            f2 = [e[0]] * e[1]
            temp.extend(f2)
    wrong = temp.copy()

    fcorr = [[y, corr.count(y)] for y in set(corr)]
    temp = []
    for e in fcorr:
        if e[0] in aux:
            f3 = [e[0]] * e[1]
            corwrong.extend(f3)
            tempdf2.append([e[0], e[1]])
        else:
            f4 = [e[0]] * e[1]
            temp.extend(f4)
    corr = temp.copy()

    # Same as before as we divided into its constituent parts.
    # print("Total Word Count :" len(corr)+len(wrong)+len(corwrong))

    fall = [[z, list_err.count(z)] for z in set(list_err)]
    fall = sorted(fall, key=lambda l: l[1], reverse=True)

    fcorwrong = [[w, corwrong.count(w)] for w in set(corwrong)]
    fcorwrong = sorted(fcorwrong, key=lambda l: l[1], reverse=True)

    fcorr = [[p, corr.count(p)] for p in set(corr)]
    fcorr = sorted(fcorr, key=lambda l: l[1], reverse=True)

    fwrong = [[q, wrong.count(q)] for q in set(wrong)]
    fwrong = sorted(fwrong, key=lambda l: l[1], reverse=True)

    ftotal = [[z, total.count(z)] for z in set(total)]
    ftotal = sorted(ftotal, key=lambda l: l[1], reverse=True)

    # Total Incorrect words constituting Pure incorrect word as well as partially correct as well as incorrect word.
    # errorall = pd.DataFrame(fall,columns=['words','count'])
    # Checking data
    # print(errorall.describe())

    # Always Correct Words and their Frequencies.
    corrall = pd.DataFrame(fcorr, columns=["words", "count"])
    # Checking Data
    # print(corrall.describe())

    # Partially correct as well as incorrect words and their Frequencies.
    corrwrongall = pd.DataFrame(fcorwrong, columns=["words", "count"])
    # print(corrwrongall.describe())
    corrwrongall = corrwrongall.merge(
        pd.DataFrame(tempdf1, columns=["words", "wrongcount"]), on="words"
    )
    corrwrongall = corrwrongall.merge(
        pd.DataFrame(tempdf2, columns=["words", "rightcount"]), on="words"
    )
    # Checking data
    # print(corrwrongall.head())
    # print(corrwrongall.describe())

    # Always Incorrect words and their Frequencies.
    erroronly = pd.DataFrame(fwrong, columns=["words", "count"])
    # Checking data
    # print(erroronly.describe())

    # All the unique words of hypothesis and their frequencies.
    totalwords = pd.DataFrame(ftotal, columns=["words", "count"])
    # Checking data
    # print(totalwords.describe())

    # Training Data Analysis

    # TXT containing training sentences(transcripts).
    TrainSentdata = pd.read_csv(
        "Data Files/Train/transcription.txt", sep="\t", names=["Id", "Text"]
    )

    # Split for validation set(exclude the validation set)
    # TrainSentdata = TrainSentdata[:16000]

    # Checking data
    # print(TrainSentdata)

    # Cleaning the Training data and converting the data into word list.
    groundtrain = list(TrainSentdata["Text"].values)
    groundtrain = [
        re.sub(
            r"[^\u0A81\u0A82\u0A83\u0A85\u0A86\u0A87\u0A88\u0A89\u0A8A\u0A8B\u0A8C\u0A8D\u0A8F\u0A90\u0A91\u0A93\u0A94\u0A95\u0A96\u0A97\u0A98\u0A99\u0A9A\u0A9B\u0A9C\u0A9D\u0A9E\u0A9F\u0AA0\u0AA1\u0AA2\u0AA3\u0AA4\u0AA5\u0AA6\u0AA7\u0AA8\u0AAA\u0AAB\u0AAC\u0AAD\u0AAE\u0AAF\u0AB0\u0AB2\u0AB3\u0AB5\u0AB6\u0AB7\u0AB8\u0AB9\u0ABC\u0ABD\u0ABE\u0ABF\u0AC0\u0AC1\u0AC2\u0AC3\u0AC4\u0AC5\u0AC7\u0AC8\u0AC9\u0ACB\u0ACC\u0ACD\u0AD0\u0AE0\u0AE1\u0AE2\u0AE3\u0AF1 ]",
            "",
            item.strip(),
        )
        for item in groundtrain
    ]

    # Finding frequencies of words which are correctly and incorrectly predicted in the Training set.
    trainlist = []
    trainwrong = []
    traincorr = []
    traincorrwrong = []

    for i in range(len(groundtrain)):
        y = groundtrain[i].replace(".", "")
        trainlist.extend(list(y.split(" ")))

    auxtrain = list(set(trainlist).intersection(set(wrong)))
    ftraintotal = [[x, trainlist.count(x)] for x in set(trainlist)]
    ftraintotal = sorted(ftraintotal, key=lambda l: l[1], reverse=True)

    temp = []
    for e in ftraintotal:
        if e[0] in auxtrain:
            f3 = [e[0]] * e[1]
            trainwrong.extend(f3)
        else:
            f4 = [e[0]] * e[1]
            temp.extend(f4)
    traincorr = temp.copy()

    ftrainwrong = [[q, trainwrong.count(q)] for q in set(trainwrong)]
    ftrainwrong = sorted(ftrainwrong, key=lambda l: l[1], reverse=True)

    ftraincorr = [[y, traincorr.count(y)] for y in set(traincorr)]
    auxtrain2 = list(set(traincorr).intersection(set(corr)))

    ftraintotal = [[w, trainlist.count(w)] for w in set(trainlist)]
    ftraintotal = sorted(ftraintotal, key=lambda l: l[1], reverse=True)

    temp = []
    for e in ftraincorr:
        if e[0] not in auxtrain2:
            f3 = [e[0]] * e[1]
            traincorrwrong.extend(f3)
        else:
            f4 = [e[0]] * e[1]
            temp.extend(f4)
    traincorr = temp.copy()

    auxtrain3 = list(set(traincorrwrong).intersection(set(corwrong)))

    ftraincorwrong = [[w, traincorrwrong.count(w)] for w in set(traincorrwrong)]

    temp = []
    for e in ftraincorwrong:
        if e[0] not in auxtrain3:
            pass
        else:
            f4 = [e[0]] * e[1]
            temp.extend(f4)
    traincorrwrong = temp.copy()

    ftraincorr = [[p, traincorr.count(p)] for p in set(traincorr)]
    ftraincorr = sorted(ftraincorr, key=lambda l: l[1], reverse=True)

    ftraincorwrong = [[w, traincorrwrong.count(w)] for w in set(traincorrwrong)]
    ftraincorwrong = sorted(ftraincorwrong, key=lambda l: l[1], reverse=True)

    # Count of total unique words and their frequencies.
    traintotalwords = pd.DataFrame(ftraintotal, columns=["words", "count"])
    # Checking Data
    # print(traintotalwords.describe())

    # Count of words in training that are always incorrectly predicted in inferencing and their frequencies.
    trainerroronly = pd.DataFrame(ftrainwrong, columns=["words", "count"])
    # Checking Data
    # print(trainerroronly.describe())

    # Count of words in training that are predicted correctly predicted as well as sometimes incorrectly predicted in testing and their frequencies.
    traincorrwrongall = pd.DataFrame(ftraincorwrong, columns=["words", "count"])
    # Checking Data
    # print(traincorrwrongall.describe())

    # Count of words in training that are predicted always correct in testing and their frequencies.
    traincorrall = pd.DataFrame(ftraincorr, columns=["words", "count"])
    # Checking Data
    # print(traincorrall.describe())

    # Saving the word level analysis:

    # Words in testing which are predicted incorrectly as well as correctly and their frequencies.
    corrwrongall.to_csv(
        "testcorrectwrongmix_" + type + model_name + ".csv", index=False
    )
    # Words in testing which are always correctly predicted and their frequencies.
    corrall.to_csv("testcorrect_" + type + model_name + ".csv", index=False)
    # Words in testing which are always incorrectly predicted and their frequencies.
    erroronly.to_csv("testwrong_" + type + model_name + ".csv", index=False)
    # Count of words in training that are predicted correctly predicted as well as sometimes incorrectly predicted in testing and their frequencies.
    traincorrwrongall.to_csv(
        "traincorrectwrongmix_" + type + model_name + ".csv", index=False
    )
    # Count of words in training that are predicted always correct in testing and their frequencies.
    traincorrall.to_csv("traincorrect_" + type + model_name + ".csv", index=False)
    # Count of words in training that are always incorrectly predicted in inferencing and their frequencies.
    trainerroronly.to_csv("trainwrong_" + type + model_name + ".csv", index=False)
    # All the unique words of hypothesis and their frequencies.
    totalwords.to_csv("testtotalwords_" + type + model_name + ".csv", index=False)
    # Count of total unique words and their frequencies.
    traintotalwords.to_csv("traintotalwords_" + type + model_name + ".csv", index=False)


def character_level(model_name, type):

    # List of Actual and Predicted words
    actpred = pd.read_csv("Actual_Predicted_" + type + model_name + "_ALLWORDS.csv")
    # Cleaning
    actpred.fillna(" ", inplace=True)
    actual = list(actpred["Actual_word"].values)
    preds = list(actpred["Predicted_word"].values)

    # Training Words
    corpus = pd.read_csv(
        "Data Files/Train/transcription.txt", sep="\t", names=["Id", "Text"]
    )
    # Testing Words
    test_corpus = pd.read_csv(
        "Data Files/Test/transcription.txt", sep="\t", names=["Id", "Text"]
    )
    # corpus = corpus[:16000]

    # All the words used in training as well as testing.
    groundcorpus = list(corpus["Text"].values)
    groundcorpus.extend(list(test_corpus["Text"].values))
    groundcorpus = [
        re.sub(
            r"[^\u0A81\u0A82\u0A83\u0A85\u0A86\u0A87\u0A88\u0A89\u0A8A\u0A8B\u0A8C\u0A8D\u0A8F\u0A90\u0A91\u0A93\u0A94\u0A95\u0A96\u0A97\u0A98\u0A99\u0A9A\u0A9B\u0A9C\u0A9D\u0A9E\u0A9F\u0AA0\u0AA1\u0AA2\u0AA3\u0AA4\u0AA5\u0AA6\u0AA7\u0AA8\u0AAA\u0AAB\u0AAC\u0AAD\u0AAE\u0AAF\u0AB0\u0AB2\u0AB3\u0AB5\u0AB6\u0AB7\u0AB8\u0AB9\u0ABC\u0ABD\u0ABE\u0ABF\u0AC0\u0AC1\u0AC2\u0AC3\u0AC4\u0AC5\u0AC7\u0AC8\u0AC9\u0ACB\u0ACC\u0ACD\u0AD0\u0AE0\u0AE1\u0AE2\u0AE3\u0AF1 ]",
            "",
            item.strip(),
        )
        for item in groundcorpus
    ]

    # Cleaning
    allwords = []
    for i in range(len(groundcorpus)):
        y = groundcorpus[i].replace(".", "")
        allwords.extend(list(y.split(" ")))

    # Finding frequencies for the whole corpus.
    fallwords = [[q, allwords.count(q)] for q in set(allwords)]
    fallwords = sorted(fallwords, key=lambda l: l[1], reverse=True)

    fdict = dict()
    for f in fallwords:
        fdict[f[0]] = f[1]

    keys = fdict.keys()

    ###Finding frequencies for actual and predicted words.
    factual = [0] * len(actual)
    fpred = [0] * len(preds)
    for i in range(len(actual)):
        if actual[i] in keys:
            factual[i] = fdict[actual[i]]
        if preds[i] in keys:
            fpred[i] = fdict[preds[i]]

    actpred = actpred.assign(count_of_Actual=factual)
    actpred = actpred.assign(count_of_Preds=fpred)

    # Checking the list of words
    # print(actpred)
    # print(actpred.describe())
    # print(actpred.info())

    ####Finding count of each alphabet from #allwords[] list.

    # Dictionary specific to language
    # Gujarati Language Dictionary to store the count of alphabets
    alphacount = {
        "\u0A81": 0,
        "\u0A82": 0,
        "\u0A83": 0,
        "\u0A85": 0,
        "\u0A86": 0,
        "\u0A87": 0,
        "\u0A88": 0,
        "\u0A89": 0,
        "\u0A8A": 0,
        "\u0A8B": 0,
        "\u0A8C": 0,
        "\u0A8D": 0,
        "\u0A8F": 0,
        "\u0A90": 0,
        "\u0A91": 0,
        "\u0A93": 0,
        "\u0A94": 0,
        "\u0A95": 0,
        "\u0A96": 0,
        "\u0A97": 0,
        "\u0A98": 0,
        "\u0A99": 0,
        "\u0A9A": 0,
        "\u0A9B": 0,
        "\u0A9C": 0,
        "\u0A9D": 0,
        "\u0A9E": 0,
        "\u0A9F": 0,
        "\u0AA0": 0,
        "\u0AA1": 0,
        "\u0AA2": 0,
        "\u0AA3": 0,
        "\u0AA4": 0,
        "\u0AA5": 0,
        "\u0AA6": 0,
        "\u0AA7": 0,
        "\u0AA8": 0,
        "\u0AAA": 0,
        "\u0AAB": 0,
        "\u0AAC": 0,
        "\u0AAD": 0,
        "\u0AAE": 0,
        "\u0AAF": 0,
        "\u0AB0": 0,
        "\u0AB2": 0,
        "\u0AB3": 0,
        "\u0AB5": 0,
        "\u0AB6": 0,
        "\u0AB7": 0,
        "\u0AB8": 0,
        "\u0AB9": 0,
        "\u0ABC": 0,
        "\u0ABD": 0,
        "\u0ABE": 0,
        "\u0ABF": 0,
        "\u0AC0": 0,
        "\u0AC1": 0,
        "\u0AC2": 0,
        "\u0AC3": 0,
        "\u0AC4": 0,
        "\u0AC5": 0,
        "\u0AC7": 0,
        "\u0AC8": 0,
        "\u0AC9": 0,
        "\u0ACB": 0,
        "\u0ACC": 0,
        "\u0ACD": 0,
        "\u0AD0": 0,
        "\u0AE0": 0,
        "\u0AE1": 0,
        "\u0AE2": 0,
        "\u0AE3": 0,
        "\u0AF1": 0,
    }

    for i in range(len(allwords)):
        for j in allwords[i]:
            try:
                alphacount[j] += 1
            except:
                pass

    alphabetcount = pd.DataFrame(alphacount.items(), columns=["guj_alphabets", "count"])

    # Checking the list of alphabets
    # print(alphabetcount.info())
    # print(alphabetcount.describe())
    # print(alphabetcount.head())

    ###Finding mismatching/interchanging single-letter diacritics, consonants and independents in predicted word w.r.t. actual words.
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

    # List of actual and predicted words and their frequencies.
    t_data = actpred.copy()

    # Cleaning
    t_data = t_data[t_data["same?"] == 0]
    t_data = t_data.reset_index(drop=True)
    t_data = t_data[["Actual_word", "Predicted_word"]]
    t_data = t_data.fillna(value="")

    diacratic = (
        gujarati_alphabet[:3] + gujarati_alphabet[-22:-21] + gujarati_alphabet[-20:-7]
    )
    independent = gujarati_alphabet[3:17]
    consonants = gujarati_alphabet[17:51]
    total = 0
    total_count_same_len = 0
    count_of_same_words = 0
    count_with_1_change = 0
    count_with_same_kind_of_character_change = 0
    words_with_dia_error = []
    words_with_inde_error = []
    words_with_con_error = []

    for i, d in t_data.iterrows():
        total += 1
        if len(d["Actual_word"]) == len(d["Predicted_word"]):
            total_count_same_len += 1
            if d["Actual_word"] == d["Predicted_word"]:
                count_of_same_words += 1
            else:
                incorrect_count = 0
                flag = 0
                for index in range(len(d["Predicted_word"])):
                    if d["Actual_word"][index] != d["Predicted_word"][index]:
                        incorrect_count += 1
                        if incorrect_count > 1:
                            flag = 1
                            break
                if flag == 0:
                    count_with_1_change += 1
                    print(d["Actual_word"], d["Predicted_word"])
                    for index in range(len(d["Predicted_word"])):
                        if d["Actual_word"][index] != d["Predicted_word"][index]:
                            if (
                                d["Actual_word"][index] in diacratic
                                and d["Predicted_word"][index] in diacratic
                            ):
                                count_with_same_kind_of_character_change += 1
                                # print(d['Actual'][index],d['Predicted'][index])
                                words_with_dia_error.append(
                                    [
                                        d["Actual_word"],
                                        d["Predicted_word"],
                                        d["Actual_word"][index],
                                        d["Predicted_word"][index],
                                    ]
                                )
                            if (
                                d["Actual_word"][index] in independent
                                and d["Predicted_word"][index] in independent
                            ):
                                count_with_same_kind_of_character_change += 1
                                # print(d['Actual'][index],d['Predicted'][index])
                                words_with_inde_error.append(
                                    [
                                        d["Actual_word"],
                                        d["Predicted_word"],
                                        d["Actual_word"][index],
                                        d["Predicted_word"][index],
                                    ]
                                )
                            if (
                                d["Actual_word"][index] in consonants
                                and d["Predicted_word"][index] in consonants
                            ):
                                count_with_same_kind_of_character_change += 1
                                # print(d['Actual'][index],d['Predicted'][index])
                                words_with_con_error.append(
                                    [
                                        d["Actual_word"],
                                        d["Predicted_word"],
                                        d["Actual_word"][index],
                                        d["Predicted_word"][index],
                                    ]
                                )

    # Checking Data
    """
  print(words_with_dia_error)
  print("__________")
  print(words_with_inde_error)
  print("__________")
  print(words_with_con_error)
  print("__________")
  #Count of words with Consonant error
  print(len(words_with_con_error))
  #Count of words with Diacritic error
  print(len(words_with_dia_error))
  #Count of words with Independent error
  print(len(words_with_inde_error))
  """

    cons_df = pd.DataFrame(words_with_con_error)
    dias_df = pd.DataFrame(words_with_dia_error)
    inde_df = pd.DataFrame(words_with_inde_error)

    # Diacratics
    freq_error_dia_actual = {}
    freq_error_dia_pred = {}
    for i, d in dias_df.iterrows():
        if d.get(2) in freq_error_dia_actual:
            freq_error_dia_actual[d.get(2)] += 1
        else:
            freq_error_dia_actual[d.get(2)] = 1

        if d.get(3) in freq_error_dia_pred:
            freq_error_dia_pred[d.get(3)] += 1
        else:
            freq_error_dia_pred[d.get(3)] = 1
    freq_error_dia_actual = {
        k: v
        for k, v in sorted(
            freq_error_dia_actual.items(), key=lambda item: item[1], reverse=True
        )
    }
    # print(freq_error_dia_actual)
    freq_error_dia_pred = {
        k: v
        for k, v in sorted(
            freq_error_dia_pred.items(), key=lambda item: item[1], reverse=True
        )
    }
    # print(freq_error_dia_pred)

    # Consonants
    freq_error_cons_actual = {}
    freq_error_cons_pred = {}
    for i, d in cons_df.iterrows():
        if d.get(2) in freq_error_cons_actual:
            freq_error_cons_actual[d.get(2)] += 1
        else:
            freq_error_cons_actual[d.get(2)] = 1

        if d.get(3) in freq_error_cons_pred:
            freq_error_cons_pred[d.get(3)] += 1
        else:
            freq_error_cons_pred[d.get(3)] = 1
    freq_error_cons_actual = {
        k: v
        for k, v in sorted(
            freq_error_cons_actual.items(), key=lambda item: item[1], reverse=True
        )
    }
    # print(freq_error_cons_actual)
    freq_error_cons_pred = {
        k: v
        for k, v in sorted(
            freq_error_cons_pred.items(), key=lambda item: item[1], reverse=True
        )
    }
    # print(freq_error_cons_pred)

    # Independents
    freq_error_inde_actual = {}
    freq_error_inde_pred = {}
    for i, d in inde_df.iterrows():
        if d.get(2) in freq_error_inde_actual:
            freq_error_inde_actual[d.get(2)] += 1
        else:
            freq_error_inde_actual[d.get(2)] = 1

        if d.get(3) in freq_error_inde_pred:
            freq_error_inde_pred[d.get(3)] += 1
        else:
            freq_error_inde_pred[d.get(3)] = 1
    freq_error_inde_actual = {
        k: v
        for k, v in sorted(
            freq_error_inde_actual.items(), key=lambda item: item[1], reverse=True
        )
    }
    # print(freq_error_inde_actual)
    freq_error_inde_pred = {
        k: v
        for k, v in sorted(
            freq_error_inde_pred.items(), key=lambda item: item[1], reverse=True
        )
    }
    # print(freq_error_inde_pred)

    ###Half-Conjugate Error Analysis
    # Actual conjugate fast i.e. the predicted word has whole alphabet and actual word has half alphabet.
    actualhalf_fast = []
    predshalf_slow = []
    for i in range(len(actual)):
        if ("\u0ACD" in actual[i] and "\u0ACD" not in preds[i]) and (
            len(actual[i]) == len(preds[i]) + 1
        ):
            actualhalf_fast.append(actual[i])
            predshalf_slow.append(preds[i])

    # Actual conjugate slow i.e. the predicted word has halp alphabet ad actual word has whole alphabet.
    actualhalf_slow = []
    predshalf_fast = []
    for i in range(len(actual)):
        if ("\u0ACD" not in actual[i] and "\u0ACD" in preds[i]) and (
            len(actual[i]) + 1 == len(preds[i])
        ):
            actualhalf_slow.append(actual[i])
            predshalf_fast.append(preds[i])

    conjunct_fast = pd.DataFrame(
        np.column_stack([actualhalf_fast, predshalf_slow]),
        columns=["actualhalf_fast", "predshalf_slow"],
    )
    conjunct_slow = pd.DataFrame(
        np.column_stack([actualhalf_slow, predshalf_fast]),
        columns=["actualhalf_slow", "predshalf_fast"],
    )
    # Checking Data
    # print(conjugate_fast)
    # print(conjugate_slow)

    # Saving the character level analysis:
    # Saving the updated list of actual and predicted words with their frequencies
    actpred.to_csv(
        "Actual_Predicted_" + type + model_name + "_ALLWORDS.csv", index=False
    )
    # Saving the Alphabet Count from #allwords[] list
    alphabetcount.to_csv(
        "microsoft_gujarati_alphabets_count_" + type + model_name + ".csv", index=False
    )
    # Count of words with Consonant error
    cons_df.to_csv("cons_df_" + type + model_name + ".csv", index=False)
    # Count of words with Diacritic error
    dias_df.to_csv("dias_df_" + type + model_name + ".csv", index=False)
    # Count of words with Independent error
    inde_df.to_csv("inde_df_" + type + model_name + ".csv", index=False)
    # Frequency of Diacritic actual/needed in one letter error.
    pd.DataFrame(freq_error_dia_actual, index=["frequency"]).to_csv(
        "freq_error_dia_actual_" + type + model_name + ".csv"
    )
    # Frequency of Diacritic predicted/found in one letter error.
    pd.DataFrame(freq_error_dia_pred, index=["frequency"]).to_csv(
        "freq_error_dia_pred_" + type + model_name + ".csv"
    )
    # Frequency of Consonants actual/needed in one letter error.
    pd.DataFrame(freq_error_cons_actual, index=["frequency"]).to_csv(
        "freq_error_cons_actual_" + type + model_name + ".csv"
    )
    # Frequency of Consonants predicted/found in one letter error.
    pd.DataFrame(freq_error_cons_pred, index=["frequency"]).to_csv(
        "freq_error_cons_pred_" + type + model_name + ".csv"
    )
    # Frequency of Independents actual/needed in one letter error.
    pd.DataFrame(freq_error_inde_actual, index=["frequency"]).to_csv(
        "freq_error_inde_actual_" + type + model_name + ".csv"
    )
    # Frequency of Independents predicted/found in one letter error.
    pd.DataFrame(freq_error_inde_pred, index=["frequency"]).to_csv(
        "freq_error_inde_pred_" + type + model_name + ".csv"
    )
    # Actual conjugate fast i.e. the predicted word has whole alphabet and actual word has half alphabet.
    conjunct_fast.to_csv(
        "gujarati_half_consonant_conjunctfast_pred_" + type + model_name + ".csv",
        index=False,
    )
    # Actual conjugate slow i.e. the predicted word has halp alphabet ad actual word has whole alphabet.
    conjunct_slow.to_csv(
        "gujarati_half_consonant_conjunctslow_pred_" + type + model_name + ".csv",
        index=False,
    )


def start_analysis(model_name, type):
    word_level(model_name, type)
    character_level(model_name, type)


if __name__ == "__main__":
    types = ["Greedy_", "Prefix_", "Prefix_LM_", "Prefix_WLM_", "Prefix_CLM_"]
    # Type of model used for hypothesis generation and decoding.
    model_name = "MODEL_NAME"
    # Select type of decoding by entering a number : 1) Greedy 2) Prefix with NO LM 3) Prefix with Both LM 4) Prefix with WLM 5) Prefix with CLM 6) Bert
    type = types[0]
    start_analysis(model_name, type)
