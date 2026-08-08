# 技术方法

本文只描述当前代码实现的方法、方程、规范选择和接口边界，不包含实验结果、性能比较或历史版本结论。

## 1. 问题定义

输入包含 $n_c$ 个基础线圈、场周期数 $N_{\rm FP}$ 和目标对称性。第 $j$ 个线圈以 16 阶实 Fourier 曲线和电流表示：

$$
\begin{aligned}
\mathbf r_j(t) &= \mathbf c_{j,0} + \sum_{k=1}^{16}
\left[\mathbf s_{j,k}\sin(2\pi k t) + \mathbf c_{j,k}\cos(2\pi k t)\right], \\
I_j &\in \mathbb R.
\end{aligned}
$$

每个线圈 token 由 $x/y/z$ 各 33 个系数和一个电流组成，共 100 维。基础线圈通过场周期旋转和 stellarator symmetry 生成完整线圈组。真空磁场由离散线段 Biot-Savart 求和得到。

快速评估输出：

- `status`：`ok`、`no_axis`、`no_surface`、`drift_rejected`、`flux_rejected`、`alpha_failed`、`branch_lost` 或 `internal_error`；
- `score`： $[0,100]$，越大越好；
- 七个分量、物理诊断量和分阶段计时。

设计目标是固定工作预算和高批量吞吐。可以批量追踪、场采样和线性代数的部分放在 GPU；拟合优先写成线性最小二乘。完整 Simsopt/Newton/DESC 验证是独立的候选接受路径。

## 2. 磁轴

以柱坐标环向角 $\phi$ 为自变量，磁力线满足

$$
\begin{aligned}
\frac{dR}{d\phi} &= R\frac{B_R}{B_\phi}, \\
\frac{dZ}{d\phi} &= R\frac{B_Z}{B_\phi}.
\end{aligned}
$$

对一个场周期 $\Delta\phi=2\pi/N_{\rm FP}$ 定义 Poincare 映射

$$
\begin{aligned}
P(R,Z) &= (R',Z'), \\
F(R,Z) &= P(R,Z) - (R,Z).
\end{aligned}
$$

磁轴截面是 $F=0$ 的固定点。实现顺序如下：

1. 在轴搜索区域建立二维初值网格，一次 GPU kernel 同时追踪全部初值一个场周期。
2. 计算 $\|F\|_2$，提取局部极小候选；候选数量和网格大小都有固定上限。
3. 对候选执行固定次数的二维 Newton 修正；主网格无解时使用更密但仍有界的后备网格。
4. 以更高精度重新计算候选残差，不接受只在低精度下成立的固定点。
5. 对 Poincare 映射做有限差分得到单周期线性化矩阵 $J=DP$。在 $\det J>0$ 时使用

   $$
   \frac{|\mathrm{tr}\,J|}{\sqrt{\det J}} < 2
   $$

   判定严格椭圆固定点；拓扑裕度只用于排序和连续质量，不改变数学存在条件。
6. 从选中固定点追踪整条轴，保存 $R_a(\phi),Z_a(\phi)$ 及其周期 Hermite 插值导数。

独立样本使用全局搜索。局部优化端点可以传入中心样本的轴提示；严格延续模式只在提示邻域内寻找同一分支，失败时返回 `branch_lost`。

## 3. 拟合不变量 $s$

在磁轴移动截面中定义无量纲坐标

$$
\begin{aligned}
X &= \frac{R - R_a(\phi)}{a}, \\
Y &= \frac{Z - Z_a(\phi)}{a}.
\end{aligned}
$$

拟合基底从二次项开始，使轴上 $s=0$ 且一阶导数为零：

$$
\begin{aligned}
s(X,Y,\phi) ={}& X^2
+ \sum_{\substack{2 \le p+q \le D \\ (p,q) \ne (2,0)}}
c_{pq0}X^pY^q \\
&+ \sum_{2 \le p+q \le D}\sum_{m=1}^{M}
X^pY^q\left[
c^{c}_{pqm}\cos(mN_{\rm FP}\phi)
+ c^{s}_{pqm}\sin(mN_{\rm FP}\phi)
\right].
\end{aligned}
$$

未知系数通过

$$
\min_{\mathbf c}\ \sum_i
\left[\mathbf B(\mathbf x_i)\cdot\nabla s(\mathbf x_i)\right]^2
+ \gamma\|D\mathbf c\|_2^2
$$

求解。 $\nabla s$ 包含移动磁轴产生的 $\partial_\phi R_a,\partial_\phi Z_a$ 项。

方程 $\mathbf B\cdot\nabla s = 0$ 是齐次的：若不指定规范， $s\equiv 0$ 是直接最小值，且任意整体缩放都等价。实现把 $X^2$ 的 $m=0$ 系数固定为 1，把它移到右端；其余系数由列缩放后的 ridge 最小二乘求出。这只固定坐标尺度，不把 $s$ 当作物理磁通。

训练点覆盖一个场周期内、磁轴周围半径 $[\rho_{\min},a]$ 的体区域；验证点独立随机取样，检查 $\mathbf B$ 与 $\nabla s$ 的夹角误差，而不是只报告训练残差。

## 4. 候选磁面与外层面选择

给定递增候选值 $s_k$，在每个 $(\theta,\phi)$ 射线上解

$$
s\!\left(R_a + r\cos\theta,Z_a + r\sin\theta,\phi\right) = s_k.
$$

射线根由二次近似初始化并用固定次数 Newton 修正。随后把整圈根作为磁力线初值，追踪一个或多个场周期，比较回归点与原环的绝对/相对漂移；同时检查有效根数、半径范围和长时间追踪。

原生评分器提供两种选择方式：

- 全局离散模式：逐级筛选固定 $s_k$，对少量候选做更高精度和长时间检查；适合相互独立的语料评估。
- 连续置信度模式：用固定短时间证据构造随 $s$ 单调的稳定置信度，在通过/失败区间内做固定次数磁通二分；适合局部优化，避免离散层级造成不必要跳变。

两种模式都受固定候选数、固定追踪步数和固定二分次数约束。完整物理评估不以这个筛选结论代替标准 Simsopt 面求解。

## 5. 从 $s$ 标定 $\psi$

$s$ 只有坐标意义。对多个截面和多个内层 $s_k$，通过截面径向 Gauss 求积计算环向磁通

$$
\begin{aligned}
\Phi_t(s_k,\phi) &= \int_{A(s_k,\phi)} B_\phi\,dR\,dZ, \\
\psi(s_k) &= \frac{1}{2\pi}\langle\Phi_t(s_k,\phi)\rangle_\phi.
\end{aligned}
$$

以无常数项低阶多项式拟合 $\psi(s)$，并要求导数符号一致。截面间相对标准差和外边界射线残差作为坐标质量诊断。之后

$$
\begin{aligned}
\nabla\psi &= \frac{d\psi}{ds}\nabla s, \\
\rho &= \sqrt{\frac{\psi}{\psi_{\rm edge}}}.
\end{aligned}
$$

外层面从通过筛选的候选中向外选择；拟合半径 $a$、候选阶梯和最终 $s_{\rm edge}$ 都必须随样本重新确定。

## 6. $\alpha$ 与 $\iota$

取顺时针几何角 $\theta$，使 $\nabla\psi\times\nabla\theta$ 与所用环向磁通约定方向一致。直场线坐标写为

$$
\begin{aligned}
\alpha &= \theta + \lambda(\rho,\theta,\phi) - \iota(\rho)\phi, \\
\mathbf B &\approx \nabla\psi\times\nabla\alpha.
\end{aligned}
$$

展开

$$
\lambda = \sum_{lmn} R_l^m(\rho)
\left[
a_{lmn}\cos(m\theta - nN_{\rm FP}\phi)
+ b_{lmn}\sin(m\theta - nN_{\rm FP}\phi)
\right],
$$

$$
\iota(\rho) = \sum_{k=0}^{K}\iota_k\rho^{2k}.
$$

在固定体采样点上求解向量最小二乘

$$
\min_{\{a,b,\iota_k\}}
\sum_i w_i\left\|
\mathbf B_i
- (1 + \lambda_\theta)\nabla\psi\times\nabla\theta
- (\iota - \lambda_\phi)\nabla\psi\times\nabla\phi
\right\|_2^2.
$$

设计矩阵对未知系数线性，采用列缩放后的 GPU QR。 $\alpha\mapsto\alpha+f(\psi)$ 不改变 $\nabla\psi\times\nabla\alpha$；实现从 $\lambda$ 基底中移除 $m=n=0$ 的纯磁通函数，包括常数项，从而固定规范。另行检查 $1+\lambda_\theta$，防止角坐标折叠。

## 7. 体采样与 QS 残差

在一个场周期内对 $\phi$ 分层取点； $\theta$ 使用确定性错位序列。设外边界射线半径为 $r_b(\theta,\phi)$，径向点取

$$
\begin{aligned}
r &= r_b\sqrt{\rho_{\min}^2 + (1 - \rho_{\min}^2)u}, \\
u &\in (0,1).
\end{aligned}
$$

使截面面积采样近似均匀，并提高靠近轴处的有效分辨率。柱坐标物理体积权重与

$$
w_i \propto R_i\,r_b(\theta_i,\phi_i)^2
$$

成正比。实现先生成过量候选，再按 $s$ 范围、边界根残差和 $R>0$ 过滤；若不足固定点预算则明确失败，不用少量有效点夸大质量。

令 $B=|\mathbf B|$，

$$
\begin{aligned}
A &= (\mathbf B\times\nabla\psi)\cdot\nabla B, \\
C &= \mathbf B\cdot\nabla B.
\end{aligned}
$$

对目标螺旋度 $(M,N)$，微分 QS 残差为

$$
f_C = (M\iota - N)A - (MG + NI)C.
$$

当前真空评估取 $I=0$，且

$$
G = \frac{\mu_0 I_{\rm link}}{2\pi},
$$

其符号随边界环向磁通。报告量是 $f_C/B^3$ 的体积加权 RMS、P95 和径向分箱；同时计算 QA、目标 QH 的正负螺旋度和 QP，避免目标只是在所有螺旋度上都同样差。

## 8. 分数的精确构成

对“越小越好”的非负误差 $x$，基本质量映射为

$$
q_\downarrow(x;\tau,p) = \frac{1}{1 + (x/\tau)^p}.
$$

对需要达到后饱和的量，使用区间 $[0,1]$ 上的 smoothstep $u^2(3-2u)$。ABI 10 默认七分量及权重为：

| 分量 | 权重 | 进入分量的量 |
|---|---:|---|
| axis | 10 | 固定点残差、椭圆拓扑裕度、截面椭圆长宽比 |
| psi | 10 | 独立点 $\mathbf B$ 与 $\nabla s$ 的夹角 P95/L2、训练残差 |
| surface | 10 | 有效逆纵横比、回归漂移、稳定层数或连续外延置信度 |
| coordinate | 10 | 截面磁通一致性、边界根残差、法向场、 $\alpha$ 拟合与单调性 |
| volume_qs | 42 | 全体/外层 $f_C/B^3$，并乘磁面大小和 QH 的 $\iota$ 因子 |
| iota | 10 | QH 时的最小 $|\iota|$；QA 不启用该门控 |
| coil | 8 | 长度、曲率、线圈间距、轴距、高阶模能量和电流尺度 |

未门控平均为

$$
\bar q = \frac{
10q_{\rm axis} + 10q_\psi + 10q_{\rm surf} + 10q_{\rm coord}
+ 42q_{\rm QS} + 10q_\iota + 8q_{\rm coil}
}{100}.
$$

体 QS 内部以 80% 全体误差和 20% 外层误差合成残差质量；磁面大小因子下限为 0.65，QH 的 $\iota$ 因子下限为 0.50。QH 总分还使用两个门控：

$$
\begin{aligned}
g_\iota &= 0.1 + 0.9q_\iota, \\
h &= \frac{e_{\rm competitor}}
{e_{\rm target}/\sqrt{M^2 + N^2} + e_{\rm competitor}}, \\
g_h &= 0.1 + 0.9q_h(h).
\end{aligned}
$$

其中竞争误差取 QA 与按螺旋度归一化 QP 的较小值； $q_h$ 由线性探索项和 $h\in[0.1,0.3]$ 的 smoothstep 组合。最终

$$
S = 100\min\!\left(1,\max\!\left(0,\bar q\,g_\iota g_h\right)\right).
$$

如果流水线提前失败，仍返回确定的状态和已完成诊断；调用者不应把失败案例的缺失物理量当作零误差。

## 9. $\alpha+\nu$ 完整面初值

快速 $\alpha$ 坐标给出接近直场线的面，但不是标准 Boozer 面。对固定 $\rho$ 构造 $\alpha$ 等值参数化并拟合 SurfaceXYZTensorFourier。随后用周期修正 $\nu$ 改善环向坐标：

$$
D\nu = (\partial_\phi + \iota\partial_\theta)\nu = g(\theta,\phi).
$$

对实 Fourier 模 $\exp[i2\pi(m\theta-nN_{\rm FP}\phi)]$，微分算子的除数为

$$
2\pi(m\iota - nN_{\rm FP}).
$$

常数模是规范自由度并被删除；接近共振的模显式跳过或正则化。修正后重新拟合谱面，并检查坐标映射 Jacobian，不能用折叠坐标作为标准求解初值。

## 10. 标准磁面与完整接受

每个候选 $\alpha+\nu$ 面依次执行：

1. Simsopt `BoozerSurface` 标准非线性最小二乘；
2. 完整 Newton 收敛；
3. 在与求解网格不同的多个稠密网格上计算相对残差和法向场 P95；
4. 检查环向绕数、法向非退化、谱面正则性和 Poincare 嵌套；
5. 按目标 $s$ 向外形成连续分支，要求包围体积随外移增长；形式收敛但跳入内分支的面不接受；
6. 从通过标准检查的面中选最大者，并保留相邻外侧失败作为边界证据；
7. 最终候选可运行 DESC，单独检查边界、嵌套性、Jacobian 和力残差分布。

`guard_boozer_surface.py` 只提供保守的逐步诊断；是否存在磁面由标准 LS/Newton 加独立验证决定。

## 11. Flow Matching

归一化后的数据 token 记为 $x$，参考噪声 $z\sim\mathcal N(0,I)$。训练采用直线概率路径

$$
\begin{aligned}
x_t &= (1 - t)z + tx, \\
u_t &= x - z, \\
t &\sim U[0,1].
\end{aligned}
$$

并最小化

$$
\mathcal L(\theta) =
\mathbb E\left[\|v_\theta(x_t,t,N_{\rm FP}) - u_t\|_W^2\right].
$$

$W$ 对几何 Fourier 系数使用 Parseval 物理曲线距离与相对标准化距离的混合权重，电流是独立特征。默认模型为 8 层、宽度 512、8 头、SwiGLU 隐层 1408 的非因果 Transformer，采用 PreNorm/RMSNorm；时间和 $N_{\rm FP}$ 作为条件。线圈 token 不使用位置编码，因此网络对基础线圈排列等变。

生成求解

$$
\begin{aligned}
\frac{dx}{dt} &= v_\theta(x,t,N_{\rm FP}), \\
t &: 0\to 1.
\end{aligned}
$$

同一 ODE 从 $t=1$ 积分到 $t=0$ 即得到逆向潜变量。数值积分方法和步数是离散映射定义的一部分；训练反演、继续优化和最终解码应保持一致，不能把不同步数的潜变量当作同一坐标。

## 12. 潜空间有限差分 Adam

目标函数为

$$
J(z) = S(F_\theta(z)),
$$

其中 $F_\theta$ 是固定步数 Flow 解码器， $S$ 是原生黑箱评分器。每步生成 $K$ 个两两正交、RMS 为 1 的方向 $u_j$，中心差分估计为

$$
\hat g_t = \frac{1}{K}\sum_{j=1}^K
\frac{J(z_t + c u_j) - J(z_t - c u_j)}{2c}u_j.
$$

分数上升使用标准 Adam：

$$
\begin{aligned}
m_t &= \beta_1m_{t-1} + (1 - \beta_1)\hat g_t, \\
v_t &= \beta_2v_{t-1} + (1 - \beta_2)\hat g_t^2, \\
z_{t+1} &= z_t + \eta\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
\end{aligned}
$$

实现中的有界稳健规则是评分接口的一部分：

- 端点携带中心磁轴提示并要求同分支；分支丢失不伪装成可比差分。
- 无效方向不进入梯度；可选择整步跳过。
- 有至少三个有效方向时，以方向差分的中位数/MAD 限制单个异常方向。
- 以滚动中位数/MAD 检查跨步梯度 RMS 和实际更新 RMS；拒绝时参数、两个矩和 Adam 步数一起保持不变。
- 更新中心无效时只尝试预先给定的有限回退比例，不做无界线搜索。
- `--resume` 保存并恢复潜变量、两个矩、步数、随机数状态、最佳状态和历史。

## 13. 数值与接口边界

- 原生 ABI 版本为 10；Python 结构体大小和版本必须与动态库一致。
- 普通独立评分默认执行全局磁轴搜索，不依赖历史；严格轴延续只用于显式传入提示的局部评估。
- 默认原生路径使用固定 100000 个体点、30000 个 $\alpha$ 拟合点和固定候选/追踪预算。配置可改，但结果必须记录完整配置。
- FP32 用于已验证的批量场计算和 QR；候选固定点和关键几何检查保留高精度复核。
- 快速分数是筛选和优化目标，不是磁面存在、标准 Boozer 解或 MHD 平衡的证明。
- 本仓库不携带模型和数据。训练、评分及完整验证都要求调用者提供来源和许可证明确的外部输入。
