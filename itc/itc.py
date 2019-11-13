
import pandas as py 
import numpy as np
import scipy as sp
import matplotlib.image as img 
from matplotlib import pyplot as plt
import time
import math
start_time = time.time()


G = [[1,0,0,1,1,0,1,0,0,0],[0,0,0,1,1,1,0,1,0,0],[0,0,1,1,1,0,0,0,1,0],[0,1,0,1,1,0,0,0,0,1]]

'''
Encoding 
'''
print('Encoded')

def encoding(ns,G=G):
    n=[int(ns[i]) for i in range(len(ns))]
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
    
    b_im=''
    enc=[]
    for i in imag:
        
        x=bin(i)[2:]
        x=(8-len(x))*'0' + x
        b_im=b_im+x
        enc.append(encoding(x[:4]))
        enc.append(encoding(x[4:]))
    Encoded.append(enc)
    
    en_i=[]
    for e in enc:
        s=''
        for j in e:
            s+=str(j)
        n=int(s,2)
        en_i.append(n)
    en_img.append(en_i)

plt.imshow(en_img,'gray')
plt.show()
#print(encoded)
    

    
'''
decodeing 
'''

def decoding(encoded):
    
    
    d=[]
    for k in encoded:
        k=np.array(k)
        temp=np.dot(G,[[k[i]]for i in range(len(k))])
        s=(sum(temp))/4
        h=[]
        for i in temp:
            if(i>=s):
                h.append(1)
            else:
                h.append(0)
        d.append(h)
        
        
    #print(len(d))
    dec=[]
    for i in range(0,len(d),2):
        
        x=d[i]+d[i+1]
        s=''
        for j in x:
            s+=str(j)
        n=int(s,2)
        dec.append(n)
    decoded.append(dec)
    #print(type(np.array(decoded)))
    
for e in Encoded:
    decoding(e)


'''   
converting decoded array to image 
'''

print('Decoded')
plt.imshow(decoded,'gray')
plt.show() 

print("time for encoding and decoding %s seconds " % (time.time() - start_time))

def PSNR(img1, img2):
    
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(1. / mse)

print("PSNR:",PSNR(np.array(decoded),np.array(image)))
