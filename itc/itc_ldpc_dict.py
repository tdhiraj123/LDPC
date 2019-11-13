import numpy as np
import matplotlib.image as img 
from matplotlib import pyplot as plt
import time
import math

start_time = time.time()

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

de=[]
endic=[]
encoded=[]
en_im=[]
print('Encoding')
for imag in image:
    
    enc=[]
    dx=[]
    imen=[]
    for i in imag:
        
        x=bin(i)[2:]
        x=(8-len(x))*'0' + x
        if(x[:4] in de):
            enc.append(endic[de.index(x[:4])])
        else:
            de.append(x[:4])
            endic.append(encoding(x[:4]))
            enc.append(endic[len(endic)-1])
            
        imen.append(int(''.join(map(str,enc[len(enc)-1])),2))
            
        if(x[4:] in de):
            enc.append(endic[de.index(x[4:])])
        else:
            de.append(x[4:])
            endic.append(encoding(x[4:]))
            enc.append(endic[len(endic)-1])
        imen.append(int(''.join(map(str,enc[len(enc)-1])),2))
        
   
    encoded.append(enc)
    en_im.append(imen)
    

plt.imshow(en_im,'gray')
plt.show()


'''
DECODING
'''

print('decoding')

dd=[]
ddec=[]
decoded=[]
#def decoding(encoded):
    
for Encoded in encoded: 
    d=[]
    present=[]
    previous=[]
    counter=0
    for k in Encoded:
        counter+=1
        if(k in dd):
            present=ddec[dd.index(k)]
        else:
            dd.append(k)
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
    
plt.imshow(decoded,'gray')
plt.show() 

print('\n')

print("time for encoding and decoding %s seconds ---" % (time.time() - start_time))
def PSNR(img1, img2):
    
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(1. / mse)


print("PSNR:",PSNR(np.array(decoded),np.array(image)))













