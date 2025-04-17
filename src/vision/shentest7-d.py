from Shen_np import *
import random,os

def gettokens(text,tnum,maxlenth=200000,mincount=2):
    biao=[]
    text=text[:maxlenth]
    for i in text:
        if i not in biao:
            biao.append(i)
    texttoken=[i for i in text]
    paichu=(" ", "\n", ",", ":", ";",
            "<", ">", "/", "?", "!",
            "=", "-", "(", ")", "，",
            "：", "；", "《", "》", "、",
            "？", "！", "，", "。", "（", "）", "“", "”",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    fqdict=dict()
    for i in range(tnum):
        previous_score=0
        maxtoken=""
        tokenp1=""
        tokenp2=""
        for j in range(0,len(texttoken)-1):
            if texttoken[j] in paichu or texttoken[j+1] in paichu:
                continue
            t=texttoken[j]+texttoken[j+1]
            if t in fqdict:
                c=fqdict[t]
            else:
                c=counttext(text,t)
                fqdict[t]=c
            if c<mincount:
                continue
            if texttoken[j] in fqdict:
                cl=fqdict[texttoken[j]]
            else:
                cl=counttext(text,texttoken[j])
                fqdict[texttoken[j]]=cl
            if texttoken[j+1] in fqdict:
                cr=fqdict[texttoken[j+1]]
            else:
                cr=counttext(text,texttoken[j+1])
                fqdict[texttoken[j+1]]=cr

            score=c**2.2/(cl*cr)
            if score>previous_score:
                tokenp1=texttoken[j]
                tokenp2=texttoken[j+1]
                maxtoken=t
                previous_score=score
        if maxtoken!="":
            biao.append(maxtoken)
        k=0
        while k<len(texttoken)-1:
            if texttoken[k]==tokenp1 and texttoken[k+1]==tokenp2:
                texttoken[k]=maxtoken
                texttoken.pop(k+1)
            else:
                k+=1
        #print(i,maxtoken)
    i=0
    while i <len(biao):
        if biao[i] not in texttoken:
            biao.remove(biao[i])
        else:
            i+=1
    biao.extend(["/unk","/end","/pad"])
    return biao

def embedding(num,dim):
    return [Ten2([random.gauss(0,0.04) for i in range(dim)]) for i in range(num)]

def embed(w,ebiao,biao):
    return ebiao[biao.index(w)]

def text2token(text:str,biao):
    y = []
    maxlen = max(len(i) for i in biao)
    lentext = len(text)
    while lentext > 0:
        cmaxlen = maxlen
        while cmaxlen > 0:
            token = text[:cmaxlen]
            if token in biao:
                y.append(token)
                break
            elif cmaxlen == 1:
                y.append("/unk")
                break
            else:
                cmaxlen -= 1
        text = text[cmaxlen:]
        lentext -= cmaxlen
    return y

def text2embed(text:str or list,biao,ebiao,pbiao):
    y=[]
    for i in range(len(text)):
        t=text[i]
        if t not in biao:
            t="/unk"
        y.append(ebiao[biao.index(t)]+pbiao[i])
    return y

def countnext(text:str,target,biao):
    '''
    输入文本和tokenize查找目标，输出查找目标下一个token的概率分布
    :param text: list[str,...]
    :param target: list[str,...]
    :return: dic{str:float,...}
    '''
    beg = 0
    dic = dict()
    allv = 0
    while True:
        ind = text.find("".join(target), beg)
        if ind == -1:
            break
        neartokens=text2token(text[ind:ind+len(target)+min(10,len(text)-ind)],biao)
        if len(neartokens) <=len(target):   # 预测最后一个token后面的token时直接返回
            return dic
        nexttoken=neartokens[len(target)]
        if nexttoken not in dic:
            dic[nexttoken] = 1
        else:
            dic[nexttoken] += 1
        allv += 1
        beg = ind + len(target)
    for i in dic:
        dic[i] /= allv
    return dic

def counttext(text,target):
    beg=0
    v=0
    while True:
        ind=text.find(target,beg)
        if ind==-1:
            break
        v+=1
        beg=ind+len(target)
    return v

def diary_split(text):
    t=text.split("\n")
    t=[i for i in t if not i.isdigit()]
    return t
    # tout=[""]
    # for i in t:
    #     if i.isdigit():
    #         tout.append("")
    #     else:
    #         tout[-1]+=i+"\n"
    # return tout

def randsenten(text:str,biao,ebiao,pbiao):
    # text_inpiece=diary_split(text)
    text_inpiece=text.split("\n")
    t=random.choice(text_inpiece)
    t_in_tokens=text2token(t,biao)
    t_in_tokens.append("/end")
    samp=[]
    for i in range(1,len(t_in_tokens)):
        x=text2embed(t_in_tokens[max(0,i-window):i],biao,ebiao,pbiao)
        y=Ten.zero(len(biao))
        # count=countnext(text,t_in_tokens[max(0,i-window):i],biao)
        # for i in count:
        #     y.data[biao.index(i)]=count[i]
        y[biao.index(t_in_tokens[i])]=1
        samp.append((x,y))
    return samp

def randsamples(text:list or str,biao,ebiao,pbiao,sentennum=5):
    samp=[]
    for i in range(sentennum):
        samp.extend(randsenten(text,biao,ebiao,pbiao))
    random.shuffle(samp)
    return samp

def argmax(t):
    return t.data.index(max(t.data))

def topk(t:list,k):
    t2=t[:]
    t2.sort(reverse=True)
    return random.choice(t2[:k])

window=10
embsize=100
class model:
    def __init__(self,biao):
        self.biao=biao
        self.ebiao=embedding(len(biao),embsize)
        self.pbiao=embedding(window,embsize)

        self.blocks=[MiniTransformer(8,embsize,window,True) for i in range(6)]
        self.f1=Linear(window*embsize,300)
        self.f2=Linear(300,len(biao))

    def forward(self,x,v=1):
        mask=None
        if len(x)>window:
            x=x[len(x)-window:]
        elif len(x)<window:
            mask=[0 if i<len(x) else 1 for i in range(window)]
            x.extend([self.ebiao[self.biao.index("/pad")]+self.pbiao[len(x)+i] for i in range(window-len(x))])

        for i in self.blocks:
            x=i(x,mask,True)
        x=Ten.connect(x)
        x=self.f1(x)
        x=(self.f2(x)*Ten([v]).expand(len(self.biao))).softmax()
        return x

    def optimize(self,k):
        for i in self.blocks:
            i.grad_descent_zero(k)
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        for i in self.ebiao:
            i.graddescent(k)
            i.zerograd()
        for i in self.pbiao:
            i.graddescent(k)
            i.zerograd()

    def train(self,times,batch,k,t):
        for i in range(times):
            aloss=0
            count=0
            while True:
                for x,y in randsamples(t,self.biao,self.ebiao,self.pbiao):
                    if count>=batch:
                        break
                    out=self.forward(x)
                    loss=Ten.nll(out,y)
                    Operator.back()
                    aloss+=loss.data[0]
                    count+=1
                if count>=batch:
                    break
            print(aloss/count)
            self.optimize(k/count)

    def run(self,text,v=1):
        em = text2embed(text2token(text,self.biao),self.biao,self.ebiao,self.pbiao)
        print(text, end="")
        while True:
            a = self.forward(em,v)
            a.data=list(a.data)
            Operator.computelist=[]
            #indx=argmax(a)
            indx = a.data.index(random.choices(a.data, weights=a.data)[0])
            #indx=a.data.index(topk(a.data,10))

            if self.biao[indx] == "/end":
                return
            print(self.biao[indx], end="")
            a = self.ebiao[indx]
            em.append(a)

    def runtime(self,text):
        import time
        em = text2embed(text2token(text, self.biao), self.biao, self.ebiao, self.pbiao)
        print(text, end="")
        while True:
            t0=time.perf_counter()
            a = self.forward(em)
            t1=time.perf_counter()
            print(t1-t0)
            a.data = list(a.data)
            Operator.computelist = []
            # indx=argmax(a)
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

if __name__ == "__main__":
    savename="test7-d"
    if savename in os.listdir():
        print("load",savename)
        Layer.loadall(savename)

    textname="300"
    if textname+".txt" in os.listdir():
        with open(textname+".txt","r",encoding="utf-8") as f:
            text=f.read()

    if textname+"biao"+".txt" in os.listdir():
        with open(textname+"biao"+".txt","r",encoding="utf-8") as f:
            biao=eval(f.read())
        print(textname+"biao","loaded")
    else:
        biao=gettokens(text,3000)
        with open(textname+"biao"+".txt","w",encoding="utf-8") as f:
            f.write(str(biao))
        print(textname+"biao","made")

    m=model(biao)
    warmup=True
    startk=0.00001
    targetk=0.005
    step=30
    for i in range(1000):
        if warmup and i<step:
            k=i*(targetk-startk)/step
        else:
            k=targetk
        m.train(1,20,k,text)
        Layer.saveall(savename)

    # m.run("",6)
