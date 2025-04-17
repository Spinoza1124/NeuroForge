from wuli import *
from Shen import *
import random,os

class env:
    def __init__(self):
        Phy.biao=[]
        self.car=Phy(1,[0,0,0],[0,0,0],r=3,color="red")
        self.stick=Phy(1,[random.gauss(0,2),0,0],[0,100,0],r=2)

    def go(self,act,t=0.01,time=4):
        for i in range(time):
            self.car.resilience(k=1000,other=self.stick)
            self.stick.force([0,-1,0])
            if act:
                self.car.force([1,0,0])
            else:
                self.car.force([-1,0,0])
            self.car.a[1]=0
            Phy.run(t)

    def getstat(self):
        return Ten([self.car.p[0], self.car.v[0],
                    self.stick.p[0], self.stick.p[1], self.stick.v[0], self.stick.v[1]])

def reward(stat):
    return 1 if stat.data[3]>90 and abs(stat.data[0])<100 else 0

def test(stat):
    return 0 if stat.data[0]>stat.data[2] else 1

class model:
    def __init__(self):
        self.f1=Linear(6,30)
        self.f2=Linear(30,30)
        self.f3=Linear(30,30)
        self.f4=Linear(30,30)
        self.f5=Linear(30,2)

    def forward(self,x):
        x=self.f1(x).relu()
        x+=self.f2(x).relu()
        x+=self.f3(x).relu()
        x+=self.f4(x).relu()
        x=self.f5(x)
        return x

    def choice(self,x):
        v = m.forward(x).data
        a=v.index(max(v))
        Operator.clean()
        return a

    def optimize(self,k=0.01):
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        self.f3.grad_descent_zero(k)
        self.f4.grad_descent_zero(k)
        self.f5.grad_descent_zero(k)

    def dcopy(self):
        c=model()
        c.f1=self.f1.dcopy()
        c.f2=self.f2.dcopy()
        c.f3=self.f3.dcopy()
        c.f4=self.f4.dcopy()
        c.f5=self.f5.dcopy()
        return c

def train(m):
    exp=[]
    batch=15
    for k2 in range(2):
        e=env()
        for k in range(6000):
            # print(v)
            s0=e.getstat()
            act=m.choice(s0) if random.uniform(0,1)>0.5 else random.randint(0,1)
            e.go(act)
            s1=e.getstat()
            exp.append([s0,act,reward(s1),s1])
            if e.stick.p[1]<0:
                break
    mold=m.dcopy()
    for k2 in range(80):
        aloss=0
        for b in range(batch):
            i=random.randint(0,len(exp)-2)
            x=exp[i][0]
            y=Ten([exp[i][2]+0.8 * max(mold.forward(exp[i][3]).data)])
            Operator.clean()
            out=m.forward(x).cut(0+exp[i][1],1+exp[i][1])
            loss=Ten.mse(out,y)
            aloss+=loss.data[0]
            Operator.back()
        # print(aloss/batch)
        m.optimize(0.0000002)

def show(m):
    e=env()
    Phy.tready()
    r=0
    for i in range(6000):
        act=m.choice(e.getstat())
        e.go(act)
        Phy.tplay()
        r+=reward(e.getstat())
        if e.stick.p[1] < 0:
            break
    print(r)

def evalm(m,n=3):
    r=0
    for j in range(n):
        e = env()
        for i in range(6000):
            act = m.choice(e.getstat())
            e.go(act)
            r+=reward(e.getstat())
            if e.stick.p[1]<0:
                break
    return r/n

savename="rlt-2-2"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
Layer.issave=False
while True:
    # show(m)
    train(m)
    print(evalm(m))
    Layer.saveall(savename)

