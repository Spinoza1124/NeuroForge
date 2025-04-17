import sys
# 将项目根目录添加到 Python 路径，以便导入自定义模块
sys.path.append("/mnt/shareEEx/liuyang/code/NeuroForge/")
# 从自定义模块导入物理引擎和神经网络相关的类/函数
from src.neuroforge.wuli import *
from src.neuroforge.Shen import *
import random, os, time

# 定义环境类
class env:
    def __init__(self):
        """初始化环境"""
        Phy.biao = []  # 清空物理引擎中的对象列表
        # 创建一个代表小车的物理对象 (质量1, 初始位置/速度0, 半径3, 红色)
        self.car = Phy(1, [0, 0, 0], [0, 0, 0], r=3, color="red")
        # 创建一个代表杆子的物理对象 (质量0.1, 随机初始x位置, 初始速度向上, 半径2)
        self.stick = Phy(0.1, [random.uniform(-5, 5), 0, 0], [0, 100, 0], r=2)

    def go(self, act, t=0.01, time=15):
        """
        执行一个动作并推进环境状态
        :param act: 动作 (0: 向左施力, 1: 向右施力)
        :param t: 物理模拟的单步时间间隔
        :param time: 执行多少个物理模拟步长
        """
        for i in range(time):
            # 计算小车和杆子之间的弹性力 (模拟连接)
            self.car.resilience(k=1000, other=self.stick)
            # 对杆子施加向下的重力
            self.stick.force([0, -2, 0])
            # 根据动作对小车施加水平力
            if act:
                self.car.force([100, 0, 0])  # 向右
            else:
                self.car.force([-100, 0, 0]) # 向左
            # 限制小车的垂直加速度 (使其保持在水平面上)
            self.car.a[1] = 0
            # 运行物理模拟一步
            Phy.run(t)

    def getstat(self):
        """
        获取当前环境的状态
        返回一个包含小车和杆子位置/速度信息的张量 (经过归一化)
        """
        return Ten([self.car.p[0] / 10, self.car.v[0] / 5,       # 小车 x位置, x速度
                    self.stick.p[0] / 10, self.stick.p[1] / 10, # 杆子 x位置, y位置
                    self.stick.v[0] / 5, self.stick.v[1] / 5])  # 杆子 x速度, y速度

    def isend(self):
        """检查回合是否结束 (杆子掉落)"""
        return self.stick.p[1] < 0

def reward(stat):
    """
    根据状态计算奖励
    如果杆子接近竖直且小车在中心区域，则奖励为1，否则为0
    """
    # stat.data[3] 是杆子的 y 位置 / 10
    # stat.data[0] 是小车的 x 位置 / 10
    return 1 if stat.data[3] > 9.5 and abs(stat.data[0]) < 5 else 0

# (此函数似乎未在主逻辑中使用)
def test(stat):
    """一个测试函数，比较小车和杆子的x位置"""
    return 0 if stat.data[0] > stat.data[2] else 1

def mase(x, y):
    """计算两个张量之间的均方误差或其平方根"""
    a = ((x - y)**2).sum()
    if a.data[0] < 1:
        return a
    else:
        return a**0.5

def clip(x, maxx, minx):
    """将张量 x 的值限制在 [minx, maxx] 范围内"""
    if x.data[0] > maxx:
        return Ten([maxx])
    elif x.data[0] < minx:
        return Ten([minx])
    else:
        return x

def minten(x, y):
    """返回两个张量中值较小的那个"""
    if x.data[0] <= y.data[0]:
        return x
    else:
        return y

def entropy(x):
    """计算概率分布 x 的熵 (用于 PPO 中的探索奖励)"""
    # H(x) = - sum(p(x) * log(p(x)))
    return Ten([-1]).expand(len(x)) * x * x.log()

# 定义策略网络 (Actor)
class model:
    def __init__(self):
        """初始化 Actor 网络结构"""
        self.f1 = Linear(6, 128) # 输入层 (状态维度6) 到隐藏层 (128)
        self.f2 = Linear(128, 2) # 隐藏层到输出层 (动作空间维度2)

    def forward(self, x):
        """前向传播，计算给定状态下的动作概率"""
        x = self.f1(x).relu() # 通过隐藏层并应用 ReLU 激活函数
        x = self.f2(x).softmax() # 通过输出层并应用 Softmax 得到概率分布
        return x

    def choice(self, x):
        """根据网络输出的概率分布随机选择一个动作"""
        v = self.forward(x).data # 获取动作概率列表
        # 根据概率加权随机选择一个动作的索引
        a = v.index(random.choices(v, v)[0])
        Operator.clean() # 清理计算图 (可能用于内存管理或梯度清零)
        return a

    def optimize(self, k=0.01):
        """使用梯度下降优化网络参数"""
        self.f1.grad_descent_zero(k) # 更新第一层权重
        self.f2.grad_descent_zero(k) # 更新第二层权重

    def dcopy(self):
        """创建模型的深拷贝"""
        c = model()
        c.f1 = self.f1.dcopy()
        c.f2 = self.f2.dcopy()
        return c

# 定义价值网络 (Critic)
class modelv:
    def __init__(self):
        """初始化 Critic 网络结构"""
        self.f1 = Linear(6, 128) # 输入层 (状态维度6) 到隐藏层 (128)
        self.f2 = Linear(128, 1) # 隐藏层到输出层 (状态价值，维度1)

    def forward(self, x):
        """前向传播，计算给定状态的价值估计"""
        x = self.f1(x).relu() # 通过隐藏层并应用 ReLU
        x = self.f2(x)       # 输出层直接输出价值估计
        return x

    def optimize(self, k=0.01):
        """使用梯度下降优化网络参数"""
        self.f1.grad_descent_zero(k) # 更新第一层权重
        self.f2.grad_descent_zero(k) # 更新第二层权重

    def dcopy(self):
        """创建模型的深拷贝"""
        c = modelv()
        c.f1 = self.f1.dcopy()
        c.f2 = self.f2.dcopy()
        return c

def show(m):
    """使用训练好的模型 m 运行环境并可视化"""
    e = env()
    Phy.tready() # 准备物理引擎可视化
    r = 0 # 总奖励
    for i in range(2000): # 最多运行 2000 步
        act = m.choice(e.getstat()) # 模型选择动作
        e.go(act) # 环境执行动作
        Phy.tplay() # 更新可视化
        # print(reward(e.getstat())) # 打印单步奖励
        r += reward(e.getstat()) # 累加奖励
        if e.isend(): # 如果回合结束
            break
        time.sleep(0.03) # 暂停以便观察
    print(r) # 打印回合总奖励

def evalm(m, n=1):
    """评估模型 m 的性能，运行 n 个回合取平均奖励"""
    r = 0 # 总奖励
    for j in range(n): # 运行 n 个回合
        e = env()
        for i in range(2000): # 每个回合最多 2000 步
            act = m.choice(e.getstat()) # 模型选择动作
            e.go(act) # 环境执行动作
            r += reward(e.getstat()) # 累加奖励
            if e.stick.p[1] < 0: # 如果杆子掉落，回合结束
                break
    return r / n # 返回平均奖励

# 定义经验回放缓冲区
class memory:
    def __init__(self, maxsize=10):
        """初始化缓冲区"""
        self.memo = [] # 存储经验轨迹的列表
        self.maxsize = maxsize # 缓冲区的最大容量 (存储的回合数)

    def experience(self, m, times=3, n=500):
        """
        使用当前策略模型 m 与环境交互，收集经验
        :param m: 当前策略模型 (Actor)
        :param times: 收集多少个回合的经验
        :param n: 每个回合的最大步数
        :return: 这些回合的平均奖励
        """
        total_ar = 0 # 记录所有收集回合的总奖励
        for t in range(times): # 收集 times 个回合
            e = env()
            exp = [] # 当前回合的经验列表
            ar = 0 # 当前回合的总奖励
            for i in range(n): # 每个回合最多 n 步
                s = e.getstat() # 获取当前状态
                out = m.forward(s).data # Actor 网络计算动作概率
                # 根据概率随机选择动作
                a = out.index(random.choices(out, out)[0])
                p = out[a] # 记录选择该动作的概率 (用于 PPO 计算)
                Operator.clean() # 清理计算图
                e.go(a) # 环境执行动作
                st = e.getstat() # 获取下一个状态
                r = reward(s) # 获取奖励 (注意：这里用的是动作前的状态 s 计算奖励)
                ar += r # 累加回合奖励
                # 存储经验元组: [状态, 动作, 奖励, 下一状态, 是否结束, 动作概率]
                exp.append([s, a, r, st, 0, p])
                # 提前结束条件 (如果步数相对于奖励过高，可能陷入不良状态)
                if ar > 0 and i / ar > 1.5: # e.isend(): # 或者使用环境的结束条件
                    exp[-1][4] = 1 # 标记回合结束
                    break
                elif e.isend(): # 检查环境的标准结束条件
                    exp[-1][4] = 1 # 标记回合结束
                    break
            self.memo.append(exp) # 将当前回合的经验存入缓冲区
            total_ar += ar # 累加总奖励
        # 如果缓冲区超过最大容量，移除最早的经验
        if len(self.memo) > self.maxsize:
            self.memo = self.memo[-self.maxsize:]
        return total_ar / times # 返回本次收集的平均回合奖励


def train(m, mv, memo, times=1, discount=0.99, lamb=0.99, eps=0.2, he=1):
    """
    使用 PPO 算法训练 Actor (m) 和 Critic (mv) 网络
    :param m: Actor 模型
    :param mv: Critic 模型
    :param memo: 经验回放缓冲区
    :param times: 每次训练前收集多少回合的新经验
    :param discount: 折扣因子 gamma
    :param lamb: GAE (Generalized Advantage Estimation) 中的 lambda 参数
    :param eps: PPO 裁剪范围 epsilon
    :param he: 熵奖励系数 (鼓励探索)
    """
    # 1. 收集经验
    ar = memo.experience(m, times) # 与环境交互，收集新经验，并获取平均奖励

    # 2. 遍历缓冲区中的每个回合经验进行学习
    for exp in memo.memo:
        aloss = 0 # 累积 Actor 损失
        alossv = 0 # 累积 Critic 损失
        ad = [] # 存储每个时间步的优势估计 (Advantage)
        gae = 0 # GAE 累加器

        # 3. 计算 GAE (从后往前遍历经验)
        for i in range(len(exp) - 1, -1, -1):
            # 获取当前状态 s, 动作 a, 奖励 r, 下一状态 s', 是否结束 done, 动作概率 p_old
            s, a, r, st, done, p_old = exp[i]
            # Critic 网络估计当前状态价值 V(s)
            v = mv.forward(s).data[0]
            # Critic 网络估计下一状态价值 V(s')
            v2 = mv.forward(st).data[0]
            # 计算 TD 误差 delta = r + gamma * V(s') * (1-done) - V(s)
            tdd = r + discount * v2 * (1 - done) - v
            # 计算 GAE: A_t = delta_t + gamma * lambda * A_{t+1} * (1-done)
            gae = tdd + lamb * discount * gae * (1 - done)
            ad.append(gae) # 存储当前时间步的优势估计
            Operator.clean() # 清理计算图
        ad.reverse() # 将优势估计列表反转，使其与经验顺序一致

        # 可选：优势标准化 (有时可以稳定训练)
        # mean = sum(ad) / len(ad)
        # std = max(1e-6, sum([(a - mean)**2 for a in ad])**0.5 / len(ad)) # Avoid division by zero
        # for i in range(len(exp)):
        #     ad[i] = (ad[i] - mean) / std

        # 4. 计算 PPO 损失并更新网络
        for i in range(len(exp)):
            # 获取经验数据
            s, a, r, st, done, p_old = exp[i]
            # Actor 网络重新计算当前状态 s 下的动作概率分布
            p = m.forward(s)
            # 获取采取动作 a 的新概率 p_new(a|s)
            pc = p.cut(a, a + 1) # .cut 可能是获取指定索引的概率值
            # Critic 网络重新计算当前状态 s 的价值 V(s)
            v = mv.forward(s)
            # 获取预先计算好的优势估计 A_t
            adv = Ten([ad[i]])

            # 计算熵 H(p) (鼓励探索)
            h = entropy(pc)
            # 计算概率比率 ratio = p_new(a|s) / p_old(a|s)
            # 使用 log 和 exp 来提高数值稳定性: exp(log(p_new) - log(p_old))
            ratio = (pc.log() - Ten([p_old]).log()).exp()

            # 计算 PPO 的 Surrogate Objective (带裁剪)
            # L_CLIP = min(ratio * A_t, clip(ratio, 1-eps, 1+eps) * A_t)
            surr = ratio * adv
            surr2 = clip(ratio, 1 + eps, 1 - eps) * adv
            # Actor 损失: L_Actor = - (L_CLIP + c_entropy * H)
            # 取负号因为我们要最大化目标函数，而优化器是最小化损失
            loss = Ten([-1]) * (minten(surr, surr2) + Ten([he]) * h)

            loss.onegrad() # 标记 Actor 损失用于反向传播
            aloss += loss.data[0] # 累加 Actor 损失

            # 计算 Critic 损失 (均方误差): L_Critic = (V(s) - (V(s) + A_t))^2 = (V(s) - V_target)^2
            # V_target = r + gamma * V(s') * (1-done)  (TD Target)
            # 这里使用了 GAE 作为 V_target 的估计: V_target ≈ V(s) + A_t
            # 注意：(v+adv).data 是为了创建一个新的 Tensor，避免直接修改 v 的计算图
            lossv = mase(v, Ten((v + adv).data))
            alossv += lossv.data[0] # 累加 Critic 损失

            # 梯度爆炸检查 (如果 Critic 损失过大，可能导致梯度爆炸)
            if abs(lossv.data[0]) > 50:
                print(f"loss-v{lossv.data[0]},梯度过高")
                Operator.clean() # 清理计算图，跳过此次更新
                return # 或者 break

            # 计算 Critic 损失的梯度 (这里假设 mase 内部处理了梯度计算)
            # lossv.onegrad() # 可能需要这一步，取决于 mase 的实现

            # 执行反向传播，计算 Actor 和 Critic 的梯度
            Operator.back()

    # 5. 更新 Actor 和 Critic 网络参数
    c = sum([len(i) for i in memo.memo]) # 计算总的样本数量
    m.optimize(0.001 / c) # 使用梯度下降更新 Actor (学习率调整)
    mv.optimize(0.002 / c) # 使用梯度下降更新 Critic (学习率调整)

    # 6. 打印训练信息
    with open("fil.txt", "a") as f: # 将训练日志写入文件
        f.write(f"{ar} {aloss/c} {alossv/c}\n") # 平均奖励, 平均Actor损失, 平均Critic损失
    print(ar, aloss / c, alossv / c) # 在控制台打印

# --- 主训练循环 ---
savename = "rlt-2-rebase-ppo-end2" # 模型保存文件名

# 检查是否存在已保存的模型，如果存在则加载
if savename in os.listdir():
    print("load", savename)
    Layer.loadall(savename) # 加载所有层的参数

# 初始化 Actor 和 Critic 模型
m = model()
mv = modelv()
Layer.issave = False # (这个标志位的作用不明确，可能控制是否在 Layer 内部自动保存)

# 初始化经验回放缓冲区，容量为 2 个回合
memo = memory(2)
count = 0 # 训练迭代次数计数器

# 无限循环进行训练
while True:
    # 执行一次训练迭代 (收集经验 -> 计算损失 -> 更新网络)
    train(m, mv, memo, discount=0.98, he=0.5) # 使用指定的超参数进行训练
    count += 1 # 增加迭代次数

    # 每 5 次迭代保存一次模型参数
    if count % 5 == 0:
        Layer.saveall(savename) # 保存所有层的参数到文件

    # 可以取消注释下面这行来可视化当前模型的表现
    # show(m)