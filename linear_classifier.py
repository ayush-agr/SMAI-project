from matplotlib import pyplot as plt
import sys,math,os,glob
from PIL import Image
import numpy as np
from numpy import * 
input_image=[]
image_matrix=[]
mylist = []


with open(sys.argv[1]) as f:
    lines = f.read().splitlines()

for i in lines:
    words=list(i.split())
    mylist.append(words)

path_store={}
dest=[]

for x in mylist:
    dest.append(x[0])

i=0
classes = []

for x in mylist:
    label=x[1]
    classes.append(x[1])
    path_store[label]=i
    i=i+1

f.close

for filename in dest:
    im=array(Image.open(filename).convert('L').resize((64,64),Image.ANTIALIAS))
    Arr=im.flatten()

    input_image.append(Arr)

image_matrix= asmatrix(input_image)
image_matrix=transpose(image_matrix)
row_mean=mean(image_matrix,axis=1)
image_matrix=image_matrix-row_mean
image_matrix=transpose(image_matrix)

U,S,V = linalg.svd(image_matrix)

V=transpose(V)

coefficient=matmul(image_matrix,V)
coefficient=coefficient[:,:32]

init=0
meanavg={}
varia={}
srtdata = sorted(path_store,key=path_store.get)

for j in srtdata:
    mnav = coefficient[init:(path_store[j]+1),:32]
    meanavg[j]=mean(mnav,axis=0)
    varia[j]=var(mnav,axis=0)
    init=path_store[j]+1

testimages = open(sys.argv[2],"rt")
lines=testimages.readlines()

dest=[]
for line in lines:
    dest.append(line.strip()) 


testimages.close

input_images=[]

for filename in dest:
    im=Image.open(filename).convert('L').im.resize((64,64),Image.ANTIALIAS)
    A=array(im)
    A=A.flatten()
    input_images.append(A)

image_matrix= asmatrix(input_images)

image_matrix=transpose(image_matrix)

image_matrix=image_matrix-row_mean

image_matrix=transpose(image_matrix)

coefficient_tr=matmul(image_matrix,V)

coefficient_tr=coefficient_tr[:,:32]

coefficient = np.matrix.tolist(coefficient)

coefficient_tr = np.matrix.tolist(coefficient_tr)


for i in coefficient:
    i.append(1.0)

for i in coefficient_tr:
    i.append(1.0)

classes = list(set(classes))
features = coefficient

# print(len(features[0]))

w = [[0 for i in range(len(features[0]))] for j in range(len(classes))]
w = np.array(w,dtype = np.float64)

ind = 0
mpclass = {}
for i in range(len(classes)):
    mpclass[classes[i]] = ind
    ind = ind + 1


# print(len(coefficient[0]))
eta = 0.01
for i in range(4000):
    for k in range(len(features)):
        temp = 0
        ind = mpclass[mylist[k][1]]
        sm  = 0
        mx = 0
        mx = np.max(matmul(w,features[k]))

        sm = np.sum(exp(matmul(w,features[k])-mx))
        # print(ind)
        # print(w[ind])
        p = exp(matmul(w[ind],features[k])-mx)/sm
        features[k] = np.array(features[k])
        temp = eta*(1-p)*features[k]
        w[ind] += temp 


for i in coefficient_tr:

    mx = -inf
    ind = 0
    
    for j in range(len(classes)):
        prod = matmul(w[j],transpose(i))
        if(prod > mx):
            mx = prod
            ind = j
    
    print(classes[ind])



