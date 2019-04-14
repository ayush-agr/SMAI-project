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

for x in sorted(mylist):
    dest.append(x[0])

i=0

for x in sorted(mylist):
    label=x[1]
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

final_ans={}

for img in coefficient_tr:
    for label in sorted(path_store, key=path_store.get):
        mn = meanavg[label]
        vr = varia[label]
        m = array(mn).flatten()
        v = array(vr).flatten()
        i = array(img).flatten()

        mat = v*math.pi
        Ab = (2*(mat))**0.5
        A = 1/Ab
        e_power = -((i-m)*(i-m))/(2*v)
        gauss=(math.e**(e_power))*A
        final_ans[label]=prod(gauss)
    
    prnt = sorted(final_ans, key=final_ans.get, reverse=True)
    prnt = prnt[0]
    print(prnt)


