from collections import Counter
from CreateDict import createDictionary
from WlmDict import wlmDictionary

def wlmOutput(string):

    res_dct = createDictionary()
    bi_dct, tri_dct, quad_dct = wlmDictionary()
    sl = list(string.split(" "))
    
    ul=[]
    bl=[]
    tl=[]
    ql=[]
    up=[]
    bp=[]
    tp=[]
    qp=[]
    final = 1
    for i in sl:
        try:
            ul.append(res_dct[i])
        except():
            ul.append(0)

    for i in range (0,len(sl)-1):
        s=[sl[i],sl[i+1]]
        stri = ' '.join([str(elem) for elem in s])
        # print(stri)
        count = bi_dct[stri]
        bl.append(count)

    for i in range (0,len(sl)-2):
        s=[sl[i],sl[i+1],sl[i+2]]
        stri = ' '.join([str(elem) for elem in s])
        # print(stri)
        count = tri_dct[stri]
        tl.append(count)

    for i in range (0,len(sl)-3):
        s=[sl[i],sl[i+1],sl[i+2],sl[i+3]]
        stri = ' '.join([str(elem) for elem in s])
        # print(stri)
        count = quad_dct[stri]
        ql.append(count)

    for v in range(0,len(ul)):
        up.append(ul[v]/2501841)
        if(len(sl)==1):
          return (ul[v]/2501841 + 1/2501841)
    # print(up)

    for v in range(0,len(bl)):
        try:
            k = (bl[v]/ul[v])
        except():
            k = 0.01
        bp.append(0.2*up[v+1]+0.8*k)
        if(len(sl)==2):
            return (0.2*up[v+1]+0.8*k + 1/2501841)

    for v in range(0,len(tl)):
        try:
            k = (tl[v]/bl[v])
        except():
            k = 0.01
        tp.append(0.1*up[v+2]+0.3*bp[v+1]+0.6*k)
        if(len(sl)==3):
            return (0.1*up[v+2]+0.3*bp[v+1]+0.6*k + 1/2501841)
    

    for v in range(0,len(ql)):
        try:
            k = (ql[v]/tl[v])
        except():
            k = 0.01
        qp.append(0.02*up[v+3]+0.08*bp[v+2]+0.3*tp[v+1]+0.6*k)
    
    for i in qp:
        final *= i
  
    return (final+1/2501841)