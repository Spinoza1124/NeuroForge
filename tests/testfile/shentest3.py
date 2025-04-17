from Shen import *
import cv2
from PIL import Image
import os,random,numpy,time

def randsample(path):
    picname=random.choice(os.listdir(path))
    path=path+"/"+picname
    pic=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    pic=pic/255.0
    l=[]
    for i in pic:
        l.extend(list(i))
    return Ten(l)

def randnoise():
    return Ten([random.gauss(0,1) for i in range(100)])

def argbool(t):
    return bool(t.data.index(max(t.data)))  # [0,1]==True

class model:
    def __init__(self):
        self.dfc1=Linear(28*28,512)
        self.dfc2=Linear(512,256)
        self.dfc3=Linear(256,2)

        self.gfc1=Linear(100,256)
        self.gfc2=Linear(256,512)
        self.gfc3=Linear(512,28*28)

    def dforward(self,inp):
        a=self.dfc1(inp).relu()
        a=self.dfc2(a).relu()
        a=self.dfc3(a)
        return a

    def gforward(self,x):
        a=self.gfc1(x).relu()
        a=self.gfc2(a).relu()
        a=self.gfc3(a)
        return a

    def doptimize(self,k):
        self.dfc1.grad_descent_zero(k)
        self.dfc2.grad_descent_zero(k)
        self.dfc3.grad_descent_zero(k)

    def goptimize(self,k):
        self.gfc1.grad_descent_zero(k)
        self.gfc2.grad_descent_zero(k)
        self.gfc3.grad_descent_zero(k)


def daccuracy(m,num):
    t=0
    for i in range(num//2):
        px=randsample("test")
        out=m.dforward(px)
        if argbool(out):
            t+=1
    for i in range(num//2):
        px=m.gforward(randnoise())
        out=m.dforward(px)
        if not argbool(out):
            t+=1
    return t/num

def gaccuracy(m,num):
    t=0
    for i in range(num):
        if argbool(m.dforward(m.gforward(randnoise()))):
            t+=1
    return t/num

def dtrain(m,batch=1,times=10,k=0.005):
    m.doptimize(0)
    for i in range(times):
        aloss=0
        for j in range(batch):
            y=Ten([0,1])
            x=randsample("mni")
            out = m.dforward(x)
            loss = Ten.mse(out, y)
            aloss += loss.data[0]
            loss.back()

            y=Ten([1,0])
            x=m.gforward(randnoise())
            out = m.dforward(x)
            loss = Ten.mse(out, y)
            aloss += loss.data[0]
            loss.back()
        print(aloss/batch)
        m.doptimize(k if aloss/batch>0.3 else k/10)

def gtrain(m,batch=1,times=10,k=0.005):
    m.goptimize(0)
    for i in range(times):
        aloss=0
        for j in range(batch):
            out=m.gforward(randnoise())
            loss=Ten.mse(m.dforward(out),Ten([0,1]))
            aloss += loss.data[0]
            loss.back()
        print(aloss/batch)
        m.goptimize(k if aloss/batch>0.3 else k/10)

def show(m):
    image=m.gforward(randnoise()).data

    im=numpy.array(image)*255
    im.resize(28,28)
    image=Image.fromarray(im)
    image.show()

def savepic(m):
    image = m.gforward(randnoise()).data
    im = numpy.clip(numpy.array(image) * 255,0,255).astype(numpy.uint8)
    im.resize(28, 28)
    image = Image.fromarray(im)
    image.save(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())+".png")

savename="test3"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()

for i in range(240):
    print("d")
    dtrain(m,batch=10,times=1,k=0.005)
    print("g")
    gtrain(m,batch=15,times=2,k=0.005)
    Layer.saveall(savename)
    savepic(m)
show(m)


