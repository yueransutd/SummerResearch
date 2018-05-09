# nothing 0
# loc 1
# org 2
# person 3
# misc 4

import copy
f = open("test_get_entity_blobs.txt","r")
text = f.readlines()
f.close()
text_len = len(text)
print(len(text))

""" Take NER Labels and change to one hot format."""
def normalize_labels(l_set):
    res= []
    for integer in l_set:
        if integer!=0:
            res.append(1)
        else:
            res.append(0)
        
    return res

""" Return range of possible indices for each entity. """
def get_entity_blobs(l_sets):
    r=[]
    res=[]
    for j in range(len(l_sets[0])): 
        sub_sum = 0
        for i in range(len(l_sets)):
            sub_sum += l_sets[i][j]
        r.append(sub_sum)
#    for l_set in l_sets:
#        r = l_set
#        i=0
    while(i<len(r)):
        if r[i]==0:
            i+=1
        else:
            j = i+1
            while r[j]!=0:
                j+=1
            res.append((i,j-1))
            i=j          
    return res
            

""" Return l_sets for scoring """
def candidates_per_entity(l_sets, entity_range):
    res_l_sets=[]
    for l_set in l_sets:
        res_l_set = [0]*len(l_set)
        res_l_set[entity_range[0]:entity_range[1]+1] = l_set[entity_range[0]:entity_range[1]+1]
        res_l_sets.append(res_l_set)
    return res_l_sets

""" Return candidate l_sets per entity in sentence. (for each tuple)""" 
def candidates_per_sentence(sentence, l_sets):
    res=[]
    for entity in get_entity_blobs(parse_sentence_labels()):
        res.append(candidates_per_entity(text, parse_sentence_labels()))
    
    return res
    
    
""" Get max, min, avg length of entity"""
def get_information(l_sets):
    list_of_len = [t[1]-t[0]+1 for t in get_entity_blobs(l_sets)]
    max_length = max(list_of_len)
    min_length = min(list_of_len)
    avg_length = round(sum(list_of_len)/len(list_of_len),3)
#    l =[]
#    for i in range(1,13):
#        l.append(0)
#        for leng in list_of_len:
#            if(leng==i):
#                l[i-1]+=1
#    return l    
    return max_length, min_length, avg_length
    
""" return list of l_sets for all sentences"""
def parse_sentence_labels():
    l_sets=[]
    i=0
    while(i<text_len):
        l_sets.append(parse_each_sentence_labels(text[i]))
        i+=2
    return l_sets
    
"""return l_set for each sentence"""
def parse_each_sentence_labels(annotated_sentence):
    words = annotated_sentence.split()
    length = len(words)
    lis = []
    li = []
    for i in range(length):
        li.append(0)
    i=0
    while(i<length):
        if(words[i]=="<ORG>" or words[i]=="</ORG>"):
            lis.append("o"+str(i))
        elif(words[i]=="<MISC>" or words[i]=="</MISC>"):
            lis.append("m"+str(i))
        elif(words[i]=="<LOC>" or words[i]=="</LOC>"):
            lis.append("l"+str(i))
        elif(words[i]=="<PER>" or words[i]=="</PER>"):
            lis.append("p"+str(i))
        i+=1    
    
    j=0
    tar = ["o", "m", "l", "p"]
    while(j<len(lis)):
        if(lis[j][0] in tar):
            a = lis[j][1:]
            start_index = int(a)-j
            b = lis[j+1][1:]
            end_index = int(b)-j-1
            
            for m in range(start_index,end_index):
                if(lis[j][0]=="l"):
                    li[m]=1
                elif(lis[j][0]=="o"):
                    li[m]=2
                elif(lis[j][0]=="p"):
                    li[m]=3
                elif(lis[j][0]=="m"):
                    li[m]=4       
        j+=2
    return li

#print(parse_each_sentence_labels(text[0]))   
#print(normalize_labels(parse_each_sentence_labels(text[0])))
#print(get_entity_blobs(parse_sentence_labels()))
print(candidates_per_entity(parse_sentence_labels(),(1, 3)))


def parse_single(sent):
    
    def is_start_tag(w):
        return True if w.startswith("<") and not w.startswith("</") else False
    
    def is_end_tag(w):
        return True if w.startswith("</") else False
    words = sent.split()
#print(parse_sentence_labels())
#print(get_information(parse_sentence_labels()))
#print(candidates_per_sentence(text,parse_sentence_labels()))

