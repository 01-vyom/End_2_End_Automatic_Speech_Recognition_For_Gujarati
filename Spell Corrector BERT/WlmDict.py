from collections import Counter

def wlmDictionary():
    with open('PATH_TO/gujdata.txt', 'r') as file:
        corpus = file.read().replace('\n', ' ')
    new_cor = corpus
    cor_ls = new_cor.split(" ")
    temp_ls = []
    for i in range(2501841-1):
        temp_ls.append(cor_ls[i]+" "+cor_ls[i+1])
    bi_dct = Counter(temp_ls)
    temp_ls = []
    for i in range(2501841-2):
        temp_ls.append(cor_ls[i]+" "+cor_ls[i+1]+" "+cor_ls[i+2])
    tri_dct = Counter(temp_ls)
    temp_ls = []
    for i in range(2501841-3):
        temp_ls.append(cor_ls[i]+" "+cor_ls[i+1]+" "+cor_ls[i+2]+" "+cor_ls[i+3])
    quad_dct = Counter(temp_ls)
    return bi_dct,tri_dct,quad_dct