from wuli import *
from Shen import *
import random,os,math,time

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

        # x=self.stick.p[0]-self.car.p[0]
        # y=self.stick.p[1]-self.car.p[1]
        # an = math.atan2(x,y)
        # anv=(self.stick.v[0]*y-self.stick.v[1]*x)/(x**2+y**2)
        # return Ten([self.car.p[0],self.car.v[0],an*3,anv*5])
        return Ten([self.car.p[0]/10, self.car.v[0]/5,
                    self.stick.p[0]/10, self.stick.p[1]/10, self.stick.v[0]/5, self.stick.v[1]/5])

def reward(stat):
    return 1 if stat.data[3]>9.5 and abs(stat.data[0])<5 else 0
    # return 1 if abs(stat.data[2])<0.23*3 else 0

class model:
    def __init__(self):
        self.f1=Linear(6,600)
        self.f2=Linear(600,2)

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
        for i in range(500):
            act = m.choice(e.getstat())
            e.go(act)
            r+=reward(e.getstat())
            if e.stick.p[1]<90:
                break
    return r/n

def train(m,n=500,discount=0.99):
    e=env()
    exp=[]
    ar=0
    for i in range(n):
        s=e.getstat()
        a=m.choice(s)
        e.go(a)
        st=e.getstat()
        r=reward(st)
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

    # mean = sum(rlist) / len(rlist)
    # sig = (sum([(r - mean) ** 2 for r in rlist]) / len(rlist)) ** 0.5
    # rlist = [(r - mean) / (sig + 0.00001) for r in rlist]

    aloss=0
    for i in range(len(exp)):
        p=m.forward(exp[i][0])
        pc=p.cut(exp[i][1],exp[i][1]+1)
        # entropy = Ten([0.01]) * (p * p.log()).sum()
        # print(p,rlist[i])
        loss=Ten([-1])*pc.log()*Ten([rlist[i]])#+entropy
        aloss+=loss.data[0]
        Operator.back()
    m.optimize(0.0002/len(exp))
    # print("aloss", aloss/len(exp))
    return ar

savename="rlt-2-re"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
maxp=300
while True:
    for i in range(10):
        p=train(m)
        if p>maxp:
            maxp=p
            Layer.saveall(savename+" "+str(p))
    Layer.saveall(savename)
