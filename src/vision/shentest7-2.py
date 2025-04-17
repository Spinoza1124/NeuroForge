from Shen import *
import random,os

def gettokens(text,tnum,maxlenth=100000):
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
            "？", "！", "，", "。", "（", "）", "“", "”")
    fqdict=dict()
    for i in range(tnum):
        fq=1
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
            if c>fq:
                tokenp1=texttoken[j]
                tokenp2=texttoken[j+1]
                maxtoken=t
                fq=c
        if fq==1:
            break
        biao.append(maxtoken)
        k=0
        while k<len(texttoken)-1:
            if texttoken[k]==tokenp1 and texttoken[k+1]==tokenp2:
                texttoken[k]=maxtoken
                texttoken.pop(k+1)
            k+=1
        #print(i,maxtoken)
    for i in biao:
        if i not in texttoken:
            biao.remove(i)
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
        y.append(ebiao[biao.index(text[i])]+pbiao[i])
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

def randsamples(text:list or str,biao,ebiao,pbiao):
    if type(text) ==str:
        text_inpiece=text.split("\n")
    else:       # in list case
        text_inpiece=text
        text="".join(text)
    t=random.choice(text_inpiece)
    t_in_tokens=text2token(t,biao)
    t_in_tokens.append("/end")
    samp=[]
    for i in range(1,len(t_in_tokens)-1):
        x=text2embed(t_in_tokens[max(0,i-window):i],biao,ebiao,pbiao)
        y=Ten.zero(len(biao))
        count=countnext(text,t_in_tokens[max(0,i-window):i],biao)
        for i in count:
            if i=="\n":
                y.data[biao.index("/end")]=count[i]     # 把回车替换为结束
                continue
            y.data[biao.index(i)]=count[i]
        samp.append((x,y))
    random.shuffle(samp)
    return samp

def argmax(t):
    return t.data.index(max(t.data))

window=10
embsize=40
class model:
    def __init__(self,biao):
        self.biao=biao
        self.ebiao=embedding(len(biao),embsize)
        self.pbiao=embedding(window,embsize)

        self.blocks=[MiniTransformer(20,embsize,window) for i in range(3)]
        self.f1=Linear(window*embsize,300)
        self.f2=Linear(300,len(biao))

    def forward(self,x):
        if len(x)>window:
            x=x[len(x)-window:]
        elif len(x)<window:
            x.extend([self.ebiao[self.biao.index("/pad")]+self.pbiao[len(x)+i] for i in range(window-len(x))])

        for i in self.blocks:
            x=i(x)
        x=Ten.connect(x)
        x=self.f1(x)
        x=self.f2(x).softmax()
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

    def run(self,text):
        em = text2embed(text2token(text,self.biao),self.biao,self.ebiao,self.pbiao)
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

if __name__ == "__main__":
    savename="test7-2-small"
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
        biao=gettokens(text,4000)
        with open(textname+"biao"+".txt","w",encoding="utf-8") as f:
            f.write(str(biao))
        print(textname+"biao","made")

    m=model(biao)
    for i in range(1000):
        m.train(1,25,0.006,text)
        Layer.saveall(savename)
    # print(near(m,"昨夜",10))

