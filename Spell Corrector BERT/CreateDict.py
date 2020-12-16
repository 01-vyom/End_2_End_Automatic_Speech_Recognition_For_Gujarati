import pandas as pd

def createDictionary():
    t_ata = pd.read_csv("PATH_TO/GujWikiCorpusCount.csv")
    res_dct = {t_ata.at[i,'words']: t_ata.at[i,'count'] for i in range(0, len(t_ata))}
    return res_dct