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
        self.c1s=[Conv(7,7,pad=False,stride_w=3,stride_h=3) for i in range(5)]
        self.c2s=[Conv(3,3) for i in range(5)]
        self.f1=Linear(8*8,10)

    def forward(self,inp):
        a0=[i(inp) for i in self.c1s]
        for i in range(len(a0)):
            for j in range(len(a0[0])):
                a0[i][j]=a0[i][j].relu()

        a1=[self.c2s[i](a0[i]) for i in range(len(self.c2s))]
        a2=a0[0]
        for j in range(len(a0[0])):
            for i in range(1,len(a0)):
                a2[j]+=a1[i][j]
            a2[j]=a2[j].relu()

        a=self.f1(Ten.connect(a2)).softmax()
        return a

    def optimize(self,k):
        for i in self.c1s:
            i.kernel.graddescent(k)
            i.kernel.zerograd()
            i.b.graddescent(k)
            i.b.zerograd()
        for i in self.c2s:
            i.kernel.graddescent(k)
            i.kernel.zerograd()
            i.b.graddescent(k)
            i.b.zerograd()
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
            loss.back()
            Operator.computelist=[]
        print(aloss/batch)
        m.optimize(k)

def testapicture(m,picpath):
    x = picture(picpath)
    out = m.forward(x)
    print(argmax(out),out)


savename="test2"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
train(m,batch=15,times=100,k=0.003)
Layer.saveall(savename)
print("accuracy:",accuracy(m,100))
testapicture(m,"ttai2.png")




