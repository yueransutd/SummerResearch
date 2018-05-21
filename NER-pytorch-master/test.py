oup = ""
lis = []
lis.append("1")
lis.append("zzzzz")
lis.append("2")

for e in lis:
    if e!="zzzzz":
        oup+= e+"\n"
    else:
        oup+="\n"
with open("suibian.txt" ,"w" ) as f:    
    f.write(oup)


