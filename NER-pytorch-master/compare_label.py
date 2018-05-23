def prepare_sentences(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        l = f.read().split()
    
    i=0
    lst = []
    sentence = []
    while i<len(l):
        word_lis = []
        word_lis.append(l[i])
        word_lis.append(l[i+1])
        sentence.append(word_lis)
        if word_lis[0]==".":
            lst.append(sentence)
            sentence =[]
        i+=2
    #print(lst[0])
    return lst
    # return list of list of list

def compare(lst1, lst2):
    psudo_label=[]
    print(len(lst1)==len(lst2))
    for i in range(len(lst1)):
        if str(lst1[i])== str(lst2[i]): 
        # list[i] is list of list (sentence)
        # if the whole sentence is the same, add to pseudo label set
        #and str(lst2[i])==str(lst3[i]):
            psudo_label.append(lst1[i])
    return psudo_label


prepare_sentences("./evaluation/temp/labeled_single_out.txt")




