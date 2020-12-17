import pandas as pd

def createDictionary():
    """
    Generate a dictionary consisting the word as key and its count in the corpus as value.
    """
    t_ata = pd.read_csv("Data Files/GujWikiCorpusCount.csv")
    res_dct = {t_ata.at[i,'words']: t_ata.at[i,'count'] for i in range(0, len(t_ata))}
    return res_dct