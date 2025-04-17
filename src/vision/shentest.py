from Shen import *
import cv2
import os,random

def randsample(path):
    picname=random.choice(os.listdir(path))
    path=path+"/"+picname
    pic=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    pic=pic/255.0
    l=[]
    for i in pic:
        l.extend(list(i))

    l2=[1 if int(picname[0])==i else 0 for i in range(10)]
    return Ten(l),Ten(l2)

def samle(path):
    pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pic = pic / 255.0
    l = []
    for i in pic:
        l.extend(list(i))

    l2 = [1 if int(path[-7]) == i else 0 for i in range(10)]
    return Ten(l), Ten(l2)

def picture(path):  #不返回标签
    pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pic = pic / 255.0
    l = []
    for i in pic:
        l.extend(list(i))
    return Ten(l)

def argmax(t):
    return t.data.index(max(t.data))

class model:
    def __init__(self):
        self.f1=Linear(28*28,64)
        self.f2=Linear(64,64)
        self.f3=Linear(64,10)

    def forward(self,inp):
        a=self.f1(inp)
        a=a.relu()
        a=self.f2(a)
        a=a.relu()
        a=self.f3(a)
        return a

    def optimize(self,k):
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        self.f3.grad_descent_zero(k)

def accuracy(m,num):
    t=0
    for i in range(num):
        px,py=randsample("train")
        out=m.forward(px)
        if argmax(out)==argmax(py):
            t+=1
    return t/num

def train(m,batch=30,times=100,k=0.005):
    for i in range(times):
        for j in range(batch):
            x,y=randsample("train")
            out=m.forward(x)
            loss=((out-y)**2).sum()
            print(loss.data[0])
            loss.back()
            Operator.computelist=[]
        m.optimize(k)

def testapicture(m,picpath):
    x = picture(picpath)
    out = m.forward(x)
    print(argmax(out))


savename="t"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
train(m,times=200)
Layer.saveall(savename)
print("accuracy:",accuracy(m,100))





