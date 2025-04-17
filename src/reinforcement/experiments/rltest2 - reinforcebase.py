from wuli import *
from Shen import *
import random,os,time

class env:
    def __init__(self):
        Phy.biao=[]
        self.car=Phy(1,[0,0,0],[0,0,0],r=3,color="red")
        self.stick=Phy(0.1,[random.uniform(-5,5),0,0],[0,100,0],r=2)

    def go(self,act,t=0.01,time=15):
        for i in range(time):
            self.car.resilience(k=1000,other=self.stick)
            self.stick.force([0,-2,0])
            if act:
                self.car.force([100,0,0])
            else:
                self.car.force([-100,0,0])
            self.car.a[1]=0
            Phy.run(t)

    def getstat(self):
        return Ten([self.car.p[0] / 10, self.car.v[0] / 5,
                    self.stick.p[0] / 10, self.stick.p[1] / 10, self.stick.v[0] / 5, self.stick.v[1] / 5])

def reward(stat):
    return 1 if stat.data[3]>9.5 and abs(stat.data[0])<5 else 0

def test(stat):
    return 0 if stat.data[0]>stat.data[2] else 1

class model:
    def __init__(self):
        self.f1=Linear(6,128)
        self.f2=Linear(128,2)

    def forward(self,x):
        x=self.f1(x).relu()
        x=self.f2(x).softmax()
        return x

    def choice(self,x):
        v = self.forward(x).data
        a=v.index(random.choices(v,v)[0])
        Operator.clean()
        return a

    def optimize(self,k=0.01):
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)

class modelv:
    def __init__(self):
        self.f1=Linear(6,128)
        self.f2=Linear(128,1)

    def forward(self,x):
        x=self.f1(x).relu()
        x=self.f2(x)
        return x

    def optimize(self,k=0.01):
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)

def show(m):
    e=env()
    Phy.tready()
    r=0
    for i in range(2000):
        act=m.choice(e.getstat())
        e.go(act)
        Phy.tplay()
        #print(reward(e.getstat()))
        r+=reward(e.getstat())
        if i>r*1.5:
            break
        time.sleep(0.03)
    print(r)

def evalm(m,n=1):
    r=0
    for j in range(n):
        e = env()
        for i in range(2000):
            act = m.choice(e.getstat())
            e.go(act)
            r+=reward(e.getstat())
            if e.stick.p[1]<0:
                break
    return r/n

def train(m,mv,n=500,discount=0.99):
    e=env()
    exp=[]
    ar=0
    for i in range(n):
        s=e.getstat()
        a=m.choice(s)
        e.go(a)
        st=e.getstat()
        r=reward(s)
        ar+=r
        exp.append([s,a,r,st])
        if len(exp)>ar*1.5:
            break
    R=0
    rlist=[]
    for i in range(len(exp)-1,-1,-1):
        R=exp[i][2]+R*discount
        rlist.append(R)
    rlist.reverse()
    print(ar)

    aloss=0
    for i in range(len(exp)):
        p=m.forward(exp[i][0])
        pc = p.cut(exp[i][1],exp[i][1]+1)
        v=mv.forward(exp[i][0])
        #entropy=(p*p.log()).sum()
        # print(p,rlist[i])
        loss=Ten([-1])*((Ten([rlist[i]])-v)*pc.log())#+Ten([0.0001])/entropy
        aloss+=loss.data[0]
        if abs(loss.data[0]) > 50:
            print(f"loss{loss.data[0]},梯度过高")
            return
        Operator.back()
    m.optimize(0.0005/len(exp))
    mv.optimize(0)
    for i in range(len(exp)):
        v = mv.forward(exp[i][0])
        loss=Ten.mse(v,Ten([rlist[i]]))
        Operator.back()
    mv.optimize(0.0005 / len(exp))
    #print("aloss", aloss / len(exp))

savename="rlt-2-rebase-3"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
mv=modelv()
while True:
    for i in range(10):
        train(m,mv)
    Layer.saveall(savename)
    # show(m)
