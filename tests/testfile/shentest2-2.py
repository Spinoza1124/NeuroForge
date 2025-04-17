from Shen import *
import cv2
import os,random

def picture(path):  #不返回标签
    pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pic = pic / 255.0
    l = []
    for i in pic:
        l.append(Ten(list(i)))
    return l

def randsample(path):
    picname=random.choice(os.listdir(path))
    path=path+"/"+picname
    l=picture(path)

    l2=[1 if int(picname[0])==i else 0 for i in range(10)]
    return l,Ten(l2)

def samle(path):
    l=picture(path)

    l2 = [1 if int(path[-7]) == i else 0 for i in range(10)]
    return l, Ten(l2)



def argmax(t):
    return t.data.index(max(t.data))

class model:
    def __init__(self):
        self.c1s=MultiConv(1,5,7,7,3,3,pad=False)
        self.c2s=MultiConv(5,1,3,3)
        self.f1=Linear(8*8,10)

    def forward(self,inp):
        a=self.c1s([inp])
        a=[func2d(i,Ten.relu) for i in a]
        a=self.c2s(a)
        a=func2d(a[0],Ten.relu)
        a=self.f1(Ten.connect(a)).softmax()
        return a

    def optimize(self,k):
        self.c1s.grad_descent_zero(k)
        self.c2s.grad_descent_zero(k)
        self.f1.grad_descent_zero(k)

def accuracy(m,num):
    t=0
    for i in range(num):
        px,py=randsample("test")
        out=m.forward(px)
        if argmax(out)==argmax(py):
            t+=1
    return t/num

def train(m,batch=30,times=100,k=0.005):
    for i in range(times):
        aloss=0
        for j in range(batch):
            x,y=randsample("train")
            out=m.forward(x)
            loss=Ten.nll(out, y)
            aloss+=loss.data[0]
            loss.onegrad()
            Operator.back()
        print(aloss/batch)
        m.optimize(k)

def testapicture(m,picpath):
    x = picture(picpath)
    out = m.forward(x)
    print(argmax(out),out)


savename="test2-2"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
train(m,batch=15,times=30,k=0.005)
Layer.saveall(savename)
print("accuracy:",accuracy(m,100))




