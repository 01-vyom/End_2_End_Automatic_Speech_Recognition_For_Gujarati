from collections import Counter

def wlmDictionary():
    """
    Function to generate all the possible n-grams upto 4.
    """

    with open('Data Files/gujdata.txt', 'r') as file:
        corpus = file.read().replace('\n', ' ')
    new_cor = corpus
    cor_ls = new_cor.split(" ")
    temp_ls = []
    total = 2501841
    for i in range(total-1):
        temp_ls.append(cor_ls[i]+" "+cor_ls[i+1])
    bi_dct = Counter(temp_ls)
    temp_ls = []
    for i in range(total-2):
        temp_ls.append(cor_ls[i]+" "+cor_ls[i+1]+" "+cor_ls[i+2])
    tri_dct = Counter(temp_ls)
    temp_ls = []
    for i in range(total-3):
        temp_ls.append(cor_ls[i]+" "+cor_ls[i+1]+" "+cor_ls[i+2]+" "+cor_ls[i+3])
    quad_dct = Counter(temp_ls)
    return bi_dct,tri_dct,quad_dct