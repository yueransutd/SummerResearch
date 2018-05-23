
'''
input: single column of word
output: four columns
'''
def extend_label():
    with open("./dataset/single_out.txt" , "r", encoding="utf8") as f:
        a= f.readlines()
        b=[]
        oup = ""
        for word in a:
        
            if len(word)>=1 and word!=" ":
                w = word[:-1]+" NNP I-NP I-ORG"
                if len(w)!= 15:
                    b.append(w)
                #else:
                    #b.append("zzzzz")
        for ww in b:
            #if ww!= "zzzzz":
            oup+= ww+"\n"
            #else:
                #oup+= "\n"
    with open("./dataset/origin_single_out.txt", "w", encoding = "utf8") as fi:
        fi.write(oup)

def extend_middle_label():
    oup = ""
    with open("./dataset/test.txt" , "r", encoding="utf8") as f:
        a= f.readlines()
    for line in a:
        new_line = " ".join([line.split()[0]+ " NNP I-NP "+line.split()[1]])
        oup+=new_line+"\n"

    with open("./dataset/new_test.txt", "w", encoding = "utf8") as fi:
        fi.write(oup)

extend_label()


