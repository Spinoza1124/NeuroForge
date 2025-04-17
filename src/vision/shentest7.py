from Shen import *
import random,os

def getwbiao(t):
    biao=["/unk","/end","/pad"]
    for i in t:
        if i not in biao:
            biao.append(i)
    return biao

def embedding(num,dim):
    return [Ten2([random.gauss(0,0.04) for i in range(dim)]) for i in range(num)]

def embed(w,ebiao,biao):
    return ebiao[biao.index(w)]

def text2embed(text:str or list,biao,ebiao,pbiao):
    y=[]
    for i in range(len(text)):
        y.append(ebiao[biao.index(text[i])]+pbiao[i])
    return y

def countnext(text:str,target):
    beg=0
    dic=dict()
    allv=0
    while True:
        ind=text.find(target,beg)
        if ind==-1:
            break
        if text[ind+len(target)] not in dic:
            dic[text[ind+len(target)]]=1
        else:
            dic[text[ind+len(target)]]+=1
        allv+=1
        beg=ind+len(target)
    for i in dic:
        dic[i]/=allv
    return dic

def randsamples(text:list or str,biao,ebiao,pbiao):
    if type(text) ==str:
        text=text.split("\n")
    t=random.choice(text)
    t2=[w for w in t]
    t2.append("/end")
    samp=[]
    for i in range(1,len(t2)-1):
        x=text2embed(t2[max(0,i-window):i],biao,ebiao,pbiao)
        y=Ten.zero(len(biao))
        count=countnext("\n".join(text),t[max(0,i-window):i])
        for i in count:
            if i=="\n":
                y.data[biao.index("/end")]=count[i]
                continue
            y.data[biao.index(i)]=count[i]
        samp.append((x,y))
    return samp

def argmax(t):
    return t.data.index(max(t.data))

window=20
embsize=30
class model:
    def __init__(self,biao):
        self.biao=biao
        self.ebiao=embedding(len(biao),embsize)
        self.pbiao=embedding(window,embsize)

        self.a1=MultiAtt(8,embsize)
        self.a2=MultiAtt(8,embsize)
        self.f1=Linear(window*embsize,300)
        self.f2=Linear(300,len(biao))

    def forward(self,x):
        if len(x)>window:
            x=x[len(x)-window:]
        elif len(x)<window:
            x.extend([self.ebiao[self.biao.index("/pad")]+self.pbiao[len(x)+i] for i in range(window-len(x))])
        x=sumchan2d([self.a1(x),x])
        x=sumchan2d([self.a2(x),x])
        x=Ten.connect(x)
        x=self.f1(x).relu()
        x=self.f2(x)
        return x.softmax()

    def optimize(self,k):
        self.a1.grad_descent_zero(k)
        self.a2.grad_descent_zero(k)
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        for i in self.ebiao:
            i.graddescent(k)
            i.zerograd()
        for i in self.pbiao:
            i.graddescent(k)
            i.zerograd()

    def train(self, times, batch, k, t):
        for i in range(times):
            aloss = 0
            count = 0
            while True:
                for x, y in randsamples(t, self.biao, self.ebiao, self.pbiao):
                    if count >= batch:
                        break
                    out = self.forward(x)
                    loss = Ten.nll(out, y)
                    Operator.back()
                    aloss += loss.data[0]
                    count += 1
                if count >= batch:
                    break
            print(aloss / count)
            self.optimize(k / count)

    def run(self,text):
        em = [embed(w, self.ebiao, self.biao) for w in text]
        print(text, end="")
        while True:
            a = self.forward(em)
            Operator.computelist=[]
            #indx=argmax(a)
            indx = a.data.index(random.choices(a.data, weights=a.data)[0])

            if self.biao[indx] == "/end":
                return
            print(self.biao[indx], end="")
            a = self.ebiao[indx]
            em.append(a)

def near(m:model,w,num):
    eb=m.ebiao[:]
    eb.sort(key=lambda x:Ten.sse(x,embed(w,m.ebiao,m.biao)).data[0])
    return [m.biao[m.ebiao.index(eb[i])] for i in range(num)]

savename="test7"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
with open("300.txt","r",encoding="utf-8") as f:
    t=f.read()
    biao=getwbiao(t)

m=model(biao)
# for i in range(1000):
#     m.train(1,15,0.01,t)
#     Layer.saveall(savename)
m.run("空山不见人 ")
