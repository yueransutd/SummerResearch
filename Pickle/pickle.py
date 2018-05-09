# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:44:23 2018

@author: jessicasutd
"""

import pickle  
class Person:  
    def __init__(self,n,a):  
        self.name=n  
        self.age=a  
    def show(self):  
        print(self.name+"_"+str(self.age))
        
#aa = Person("JGood", 2)   
#f=open('p.txt','wb')  
#pickle.dump(aa,f,0)  
#f.close()  
#del Person 

#f=open('p.txt','rb')  
#bb=pickle.load(f)  
#f.close()  
#bb.show()
dic = {
       "apple": 1,
       "pear": 2,
       "hh": 3}
f=open('a.txt','wb') 
pickle.dump(dic,f)  
f.close()  

f=open('a.txt','rb')  
bb=pickle.load(f)  
f.close()  
print(bb["apple"])
print(bb["pear"])
print(bb["hh"])






