import matplotlib.pyplot as plt
import os
import numpy

#import locale
#locale.getdefaultlocale()
#import locale

#locale.getpreferredencoding(do_setlocale=False)


#for sheet in os.listdir('./backup_all_sensors/'):
    #if not sheet.endswith('.csv'):
        #continue
    
data = open('./backup_all_sensors/AIT201.csv', 'r')
y=[]
x=[]
count=0
for index, value in enumerate(data):
    try:
        y.append(float(value))
        prev_value = y[count]
        x.append(count+1)
        count+=1
    except:
        if value == '\n':
            print(count)
            #count+=1
            #y.append(prev_value)
            continue


plt.plot(x,y)
plt.show()

