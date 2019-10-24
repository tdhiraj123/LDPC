
import pandas as py 
import numpy as np
import scipy as sp
import matplotlib.image as img 
from matplotlib import pyplot as plt
import time

start_time = time.time()
print('Encoding')
G = [[1,0,0,1,1,0,1,0,0,0],[0,0,0,1,1,1,0,1,0,0],[0,0,1,1,1,0,0,0,1,0],[0,1,0,1,1,0,0,0,0,1]]

'''
Encoding 
'''

def encoding(ns):
    n=[int(ns[i]) for i in range(len(ns))]
    G = [[1,0,0,1,1,0,1,0,0,0],[0,0,0,1,1,1,0,1,0,0],[0,0,1,1,1,0,0,0,1,0],[0,1,0,1,1,0,0,0,0,1]]
    G=np.array(G)
    Gt=[[G[j][i] for j in range(len(G))] for i in range(len(G[0]))] 
    #n=list(map(int,input('input:').split()))
    nt=[[n[i]] for i in range(len(n))] 
    e=np.dot(Gt,nt)
    encoded_message=[i[0]%2 for i in e]
    return encoded_message


file_name='Lena.jpg'
image = img.imread(file_name)
#print(len(image[0]))

en_img=[]
decoded=[]
Encoded=[]
for imag in image:
    
    enc=[]
    en_i=[]
    for i in imag:
        
        x=bin(i)[2:]
        x=(8-len(x))*'0' + x
        enc.append(encoding(x[:4]))
        enc.append(encoding(x[4:]))
        en_i.append(int(''.join(map(str,enc[len(enc)-2])),2))
        en_i.append(int(''.join(map(str,enc[len(enc)-1])),2))
        
    Encoded.append(enc)
    en_img.append(en_i)

plt.imshow(en_img,'gray')
plt.show()
#print(encoded)
    
    
'''
decodeing 
'''
print('Decoding')

for Encoded in encoded: 
    d=[]
    present=[]
    previous=[]
    counter=0
    for k in Encoded:
        counter+=1
        k=np.array(k)
        temp=np.dot(G,[[k[i]]for i in range(len(k))])
        s=(sum(temp))/4
        h=[]
        for i in temp:
            if(i>=s):
                h.append(1)
            else:
                h.append(0)
        ddec.append(h)
        present=h
        if(counter%2==0):
            d.append(int(''.join(map(str,previous+present)),2))
        previous=[ele for ele in present]
    decoded.append(d) 



'''   
converting decoded array to image 
'''


plt.imshow(decoded,'gray')
plt.show() 

print('\n')

print("--- %s seconds ---" % (time.time() - start_time))
