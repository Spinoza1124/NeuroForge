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
        self.f1=Linear(28*28,20)
        self.f2=Linear(20,20)
        self.f22=Linear(20,20)
        self.f23=Linear(20,20)
        self.f24=Linear(20,20)
        self.f3=Linear(20,10)

    def forward(self,inp):
        a=self.f1(inp).relu()
        b=self.f2(a).relu()
        a+=b
        b=self.f22(a).relu()
        a+=b
        b=self.f23(a).relu()
        a+=b
        b=self.f24(a).relu()
        a+=b
        a=self.f3(a)
        a=a.softmax()
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


savename="t4"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
train(m,times=200)
Layer.saveall(savename)
print("accuracy:",accuracy(m,100))





