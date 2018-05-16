with open("./dataset/single_out.txt" , "r", encoding="utf8") as f:
    a= f.readlines()
    b=[]
    oup = ""
    for word in a:
        
        if len(word)>=1 and word!=" ":
            w = word[:-1]+" NNP I-NP I-ORG"
            if len(w)!= 15:
                b.append(w)
    for ww in b:
        oup+= ww+"\n"
with open("./dataset/new_single_out.txt", "w", encoding = "utf8") as fi:
    fi.write(oup)