# Notes_On Representing Chemical Environment



## Section1：Introduction



### Representation

- 合适的原子环境的representation是现代凝聚态物理和计算化学当中很重要的概念
  - representation在结构搜索当中，每个structure在数值上是依赖于准确的初始条件和搜索路径的，因此分辨它们之间的等价性和相似性很重要。而这是依赖于representation的。

- The representation 是以descriptor的形式出现的
- descriptor是一些关于原子位置的实值函数的元组，函数包括了键长、键角等等
- 在笛卡尔坐标系当中准确描述每个原子的位置以提供一个原子结构的简单、明确的description是不合适的
  - 在这种操作下，坐标序列是随件排列的，两个看似不同的list of coordinates可能会通过旋转、反射、平移等方式互相转换。也就是说两个看似不同的descriptor实际上可能描述的是同一个结构。
  - **什么是一个好的representation**：在置换、旋转、反射、平移等操作下保持不变的representation，而同时也能够保证和笛卡尔表示相同的正确性

- 例子：
  - 将potential energy surfaces(PESs) & electrostatic multipole  moment surface和由第一性原理产生的数据进行拟合，是计算化学几十年来的重要工作
  - 在对a small group原子建模的时候，会使用成对原子的间距或是间距的倒数、指数
    - 这种做法在原子总数固定的时候是很有效的
    - 问题：当对原子进行置换时，上述的做法并不能描述整个系统的对称性
  - Braams&Bowman在**补救**这一缺点的时候采用了**使用原子间距的多项式**的方法$\Rightarrow$ 每一项都是在置换操作当中不变的
    - 10个原子以下的代码已经给出
    - 该方法仅限于固定原子总数的情况

### Completeness of descriptor

- A system of invariant descriptors $q_1,q_2,\cdots,q_M$ is said to be **complete** if it **uniquely determines** the atomic environment,up to symmetries.
  - 注意，M是系统的维度
  - descriptor的uniqueness 必须要考虑到symmetry的影响，避免出现两个不同的descriptor能够通过symmetry互相转换的情况
  - 完备的descriptor 和真正不同的原子环境之间是存在一一映射的。
- It is said to be overcomplete if it contains spurious descriptors in the sense that a proper subset of $\{q_1,q_2,\cdots,q_m\}$is  complete.
  - descriptors和真正不同的原子环境之间是多对一的映射



### Requirements and Purposes

- 为了能够产生原子间势能并且有能力描述多种不同的条件，一个原子附近对它能量有贡献的**其他原子数必须是变化的**，同时要能够保证symmetry-invariant descriptors remain continuous and differentiable。
- 在实际操作中，原子总数虽然变化了，但为了能够更好地实现function fitting，更符合实际的做法是让总维度M保持不变$\Rightarrow $没有一种传统的方法能够实现这个准则。
  - 最近提出了一些新的方法能够实现这个目的
  - 很难判断哪一种方法是最优的，下面将从构造good descriptor的角度出发探讨这个问题



## Section2：Potential Energy Surface Fitting

> We briefly recall how potential energy surfaces are constructed using invariant descriptors

- Potential Energy Surface(PES)

  - 实际上就是原子间势之和

- 总能量的短程部分可以被分解为原子间能量之和
  $$
  E_{short} = \sum_n \epsilon(q_1(n),\cdots,q_M(n))
  $$
  其中n是遍历所有的原子数，M为我们选用的descriptor的维度，$\epsilon$是第n个原子对短程能量的贡献

- $\epsilon$的求解是核心问题：

  - Traditionally, such atomic energy functions are defined in closed form.

  - Recently, there has been a lot of interest in using more flexible, nonparametric PESs.

    - 大致思想：利用第一性原理计算得到的数据，基于a certain set $\{\vec{q}^{(1)},\cdots,\vec{q}^{(N)}\}$进行拟合：实际上就是一个映射，构建起原子结构和第一性原理计算得到的数据之间的关系
    - 最简单的方法：直接线性拟合
    - 更为复杂的方法
      - 利用神经网络(NN)
      - 利用高斯近似势（GAP）

  - 这些方法最终都是得到这样一个形式
    $$
    \epsilon (\vec{q}) = \sum_{k=1}^N \alpha_k K(\vec{q},\vec{q}^{(k)})
    $$

    - N: the number of training configurations, namely, the total atoms number considered.
    - K: Kernel function, a fixed (nolinear) function, 作用是描述$\vec{q}$和$\vec{q}^{(k)}$两个原子环境的相似程度。

    因此，对于这样的一种方法，the choice of **descriptor $\vec{q}^{(k)}$** and the choice of **kernel K** are critical ingredients to obtaining good quality PESs.(尤其是$\vec{q}$在symmtry下保持的不变性)

    

### kernel K

- **K的数学特征**：大体上，可以认为K是一个协方差函数

  - 协方差：描述二者之间的关联程度，如果$X$和$Y$是统计独立的，则$E(X\cdot Y) = E(X)\cdot E(Y)$，协方差为0
    $$
    COV(X,Y) = E((X-E(X))(Y-E(Y)))=E(X\cdot Y ) - E(X)E(Y)
    $$

  - 方差：
    $$
    VAR(X) = E((X-E(X))^2) = E(X^2)-E(X)^2
    $$
    方差是协方差的一种特殊情况。
    
  - 可以定义协方差的相关性$\eta$
    $$
    \eta = \frac{COV(X,Y)}{\sqrt{VAR(X)\cdot VAR(Y)}}
    $$

  - 协方差矩阵：分别为m与n个标量元素的列向量随机变量X和Y，二者对应的期望值向量为$\mu,\nu$，则这两个变量之间的协方差定义为$m\times n$矩阵
    $$
    COV(X,Y) = E((X-\mu)(Y-\nu)^T)\Rightarrow  \text{a matrix with }m \times n
    $$

- **K的性质**：

  - 对称性：$K(\vec{q},\vec{q'}) = K(\vec{q'},\vec{q}) $
  - 正定性:  for any nonzero vector $\vec{\alpha}$ of coefficients, $\sum_k \sum_l \alpha_k \alpha_l K(\vec{q}^{(k)},\vec{q}^{(l)}) > 0$

- **K的两种常见取法**：

  - the dot-product(DP) kernel:
    $$
    K_{DP}(\vec{q},\vec{q'}) = \sum_j q_j q_j'^{(k)}
    $$
    在DP Kernel的取法之下，我们有：
    $$
    \epsilon(q) = \sum_{k=1}^N \alpha_k \sum_j q_j q_j'^{(k)} = \sum_j q_j \sum_{k=1}^N\alpha_k q_j'^{(k)} = \sum_j q_j \beta_j = \vec{q}\cdot \vec{\beta}
    $$
    在这个过程当中引入了$\vec{\beta}$。

  - 对于有$N_H$层隐藏层的NN，原子能量函数$\epsilon $由下面的式子给出
    $$
    \epsilon(\vec{q}) = b  + \sum_{j=1}^{N_H}v_j h(\vec{q},\vec{u}_j)
    $$
    其中，b是偏置函数，$\vec{v}$是单位权重，h是transfer function，$\{\vec{u}_j\}_j^{N_H}$是归一化参数。上式得到的过程没有细讲，在参考文献37当中。

    - 更关键的在于，使用上式时，我们也可以将能量函数转化为用Kernel写的形式：例如，当$h(\vec{q},\vec{u}) = \tanh(u_0 +\sum u_j q_j)$时，相应的Kernel可以写作
      $$
      K_{NN}(\vec{q},\vec{q'}) \sim -|\vec{q}-\vec{q'}|^2 + \text{const}
      $$

    
    
    
    
    
  
- K是有各种取法的，只需要满足上述的数学特征即可。





## Section3：Descriptor

- The descriptor's applications:

  - representing the geometry of an entire molecule
  - describing the neighborhood of an atom perhaps within a finite cutoff distance

  尽管上述两种cases是紧密联系的，但是很多时候descriptor只能满足其中之一。

  - 下面主要强调能够**满足第二点**：准确描述原子周围环境的功能，而对于第一点，在某些情况之下是可以泛化得到的

### Weyl矩阵

- 以中心原子为参照，N个相邻原子的位置矢量是$\{\vec{r}_1,\cdots,\vec{r}_N\}$

  - 从直观上面，有一种很直接的方法能够用来比较两个原子结构的相似程度：
    $$
    \Delta = \min_\hat{R}\Delta(\hat{R}) =\min_{\hat{R}} \sum_i^N |\vec{r_i}-\hat{R}\vec{r}'_i|
    $$
    其中$\hat{R}$是任意的旋转操作（也包括了反射）。

    - 缺点：因为它是不可导的，不能用来参数化势能平面(PES)

  - Weyl矩阵是：<img src="C:\Users\season\AppData\Roaming\Typora\typora-user-images\image-20211010093820554.png" alt="image-20211010093820554" style="zoom: 67%;" />

    该矩阵满足：在旋转，反射和平移操作下，矩阵当中的矩阵元是不变的

    - 但是：在permutation下，原子的编号会发生改变，因此$\Sigma$会发生变化，比如，1和2互换位置：<img src="C:\Users\season\AppData\Roaming\Typora\typora-user-images\image-20211010095239656.png" alt="image-20211010095239656" style="zoom:67%;" />

  - 采用Weyl矩阵的时候：可以用下述的reference distance metric
    $$
    d_{ref} = \min_P || \Sigma - P\Sigma'P^T ||
    $$
    其中，$P$是permutation矩阵。

    上面表达式的意义是，对于两个Weyl矩阵，可以通过$d_{ref}$来比较两个矩阵$\Sigma$和$\Sigma'$之间的相似程度。$d_{ref}$越小，说明两个结构越相似。

    - 这个方法的缺点：在空间上不可导

#### 从Weyl矩阵推导不变量

- 仔细观察互换编号1和2之后的矩阵，可以发现，实际上就是第一行和第二行互换位置，第一列和第二列互换位置。初等行列变换之下**矩阵的特征值**是不变的。
- 但采用特征值描述必然是not complete:
  - 原因：the dimensionality of the configuration space of N neighbor atoms is 3N-3（不考虑平移对称性）
    -  而特征值只有N个
    - 同时让特征值随着近邻原子数连续可导也是一个难题

#### A. Bond-order parameter

- 在推导原子环境的连续的不变表示时，第一步需要引入Bond-order parameter这一个参数

- Definition：the atomic neighbor density function associated with a given atom
  $$
  \rho(\vec{r}) = \sum_i w_{Z_i}\delta(\vec{r}-\vec{r_i})
  $$

  - $w_{Z_i}$是权重因子，和第i个原子的原子种类相关

  - i遍历了$\vec{r}$处原子周围的近邻原子，作为$\rho(\vec{r})$

    - 利用cutoff function来决定哪些原子算作在$\vec{r}$近邻的原子

  - $\rho$的好处在于对于permutation操作是不变的：permutation操作只是变换了求和过程中的顺序

  - 为了简化下面的步骤，我们省略径向信息，只考虑角度的影响，因此可以用球谐函数展开：
    $$
    \rho(\hat{\vec{r}}) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l}c_{lm}Y_{lm}(\hat{\vec{r}})
    $$
    其中，$\hat{\vec{r}} = \frac{\vec{r}}{r}$。$\rho(\hat{\vec{r}})$是$\rho({\vec{r}})$在单位球面上的投影。显然这个函数是和$SO(3)$群有关的。

    - $c_{lm} = \bra{\rho}\ket{Y_{lm}} = \sum_i Y_{lm}(\hat{\vec{r_i}})$

##### second-order bond-order parameters & third-order bond-order parameters

- 定义一个新的量$Q_{lm}$，它正比于$c_{lm}$
  - $Q_{lm} = \frac{1}{N}\sum_l Y_{lm}(\hat{\vec{r_i}})$
  - $Q_{lm}$相当于是给定中心原子的情况下得到的结果，而全域的$Q_{lm}$需要对所有的原子求和
  - 全域：$\overline{Q}_{lm} = \frac{1}{N_b}\sum_{i,i'}Y_{lm}(\hat{\vec{r}_{ii'}})$
    - $N_b$ is the total number of atom pairs.
    - $\vec{r}_{ii'}$ 是从atom i 指向atom i‘的向量
- <img src="C:\Users\season\AppData\Roaming\Typora\typora-user-images\image-20211010105959173.png" alt="image-20211010105959173" style="zoom:80%;" />



​    在括弧当中的因子是Wigner 3jm symbol。正交处理如下：![image-20211010110257946](C:\Users\season\AppData\Roaming\Typora\typora-user-images\image-20211010110257946.png)

##### Comments：

- Bond-order parameters 最初是为了研究液体和玻璃当中的序而引入的，但很快就有了广泛的应用。
- 它们在研究量子液体融化、团簇的自由能，核物理以及模拟相变、产生原子间相互作用势能方面都有用处。



#### B.Power spectrum

​	***以下内容涉及到很多群论方面的知识，如果确实有学习的这部分内容的需要，等把群论整理完之后再回来。***







## Section4：A similarity measure between atomic environments

- 在上文当中已经提到了，我们构建descriptor的方法的自由度是很大的，包括了：

  - angular band limit
  - the radial basis
  - which subset of the basis elements are actually used

- Kernel K是建立起descriptor的关键



### SOAP的推导过程  

因此，这里提出一种新的方法，只需要建立K即可，而绕开descriptor的内容。

  - 先建立**similarity**：
    $$
    S(\rho,\rho') = \int \rho(\vec{r})\rho'({\vec{r}}) d\vec{r}
    $$

    - 这是两个原子环境之间的相似性，$\rho$和$\rho'$是两种原子环境对应的环境密度
    - 它实际上就是两个密度的内积
    - 它是满足置换不变性的

  - 在**similarity**的基础上，建立**rotationally invariant similarity kernel**:
    $$
    k(\rho,\rho') = \int |S(\rho,\hat{R}\rho')|^n d\hat{R} = \int d\hat{R} |\int \rho(\vec{r})\rho'({\hat{R}\vec{r}}) d\vec{r}|^n
    $$

    - 这里用的Kernel的定义是：**dot-product covariance Kernel**
    
  - 接下来可以利用**rotationally invariant similarity kernel**推导出**SOAP kernel**
    $$
    K(\rho,\rho') = (\frac{k(\rho,\rho')}{\sqrt{k(\rho,\rho)k(\rho',\rho')}})^\zeta
    $$
    

    - $\zeta$可以认为是用于定义**SOAP kernel**的**敏感度的指标**
      
    - 在具体计算中的技巧：$\rho$在定义当中是有Dirac-delta函数在内的，这破坏了连续性。替代的方法是利用Gaussians，并能利用球谐函数展开：
      $$
      exp(-\alpha|\mv-{\mv}_i|^2) = 4\pi exp[-\alpha(r^2+r_i^2)]\sum_{lm}\tau_l(2\alpha rr_i)Y_{lm}(\hat{\vec{r}})Y_{lm}^*(\hat{\vec{r_i}})
      $$
      其中的$\tau_l$是modified第一类球贝塞尔函数。
      $$
      \DeclareMathOperator*{\md}{\text{d}}
      \DeclareMathOperator*{\mv}{\boldsymbol{r}}
      
      \rho(\mv) = \sum_i \exp({-\alpha|\mv-{\mv}_i|^2}) = \sum_i\sum_{lm} c_{lm}^i(r)Y_{lm}(\hat{\mv})\\
      \text{where}\quad 
      c_{lm}^i(r) = 4\pi \exp[-\alpha(r^2+r^2_i)]\tau_l(2\alpha r r_i)Y_{lm}^*(\hat{\mv}_i)
      $$
      
    - 有了$\rho$的表达式，就可以求**similarity**，在这个过程当中要充分地利用**球谐函数的正交性**。
    
    - 有**similarity**的表达式，就可以求**SOAP Kernel**，在这个过程当中要充分**利用Wigner matrices的正交性**。
    
    
      - 同时还有一个问题：n的选取。n为正整数，无论取多少都是可以在数学上表示出来的。

        - n=1丢失了角度的信息，$n\ne 1$是能够保留kernel的角度信息
    

### 利用径向基函数改进SOAP算法

- 上面的计算过程中，在求**similarity**时会遇到这样的计算过程：<img src="C:\Users\season\AppData\Roaming\Typora\typora-user-images\image-20211012094334700.png" alt="image-20211012094334700" style="zoom:67%;" /><img src="C:\Users\season\AppData\Roaming\Typora\typora-user-images\image-20211012094255958.png" alt="image-20211012094255958" style="zoom:67%;" />
  - 每一个$\widetilde{I}_{mm'}^l(\alpha,r_i,r_i')$都需要重新计算each pair of neighbors,这对于大量neighbors的情况是非常麻烦的。

- 解决方案：在对$\rho$ 进行展开的时候，不要把径向信息放到系数当中，而是再引入一个正交完备的径向基函数，这样径向信息也可以利用正交性的方式处理：

  - $$
    \rho(\mv) = \sum_i \exp({-\alpha|\mv-{\mv}_i|^2}) = \sum_{nlm} c_{nlm}g_n(r)Y_{lm}(\hat{\mv})\\
    \text{where}\quad 
    $$

    - 其中径向函数$g_n(r)$是一组正交完备基

  - 这样在求Kernel的时候，不用再计算复杂的径向关系

  - 文中给出了$n=2\quad \text{or}\quad 3$时Kernel的表达式：<img src="C:\Users\season\AppData\Roaming\Typora\typora-user-images\image-20211012100020058.png" alt="image-20211012100020058" style="zoom:67%;" />

- 总结该部分内容：
  - 直接定义了一个SOAP **similariy Kernel**，绕过了descriptor的计算
  - 抓住了问题的核心，也就是Kernel。用$\rho$建立similarity以表示这个Kernel，这样的思想其实也包含着密度泛函理论的深刻思想在其中：**用电子密度表示一切量**



## Section5：Numerical Results

> 利用数值实验检验SOAP的有效性（faithfulness of the represnetation）

- representation的有效性的定义：在某种representation下，没有不同的结构指向同一个descriptor，这说明这个表示是有效的

- 数值检验中的难点和方案：
  - 几乎不可能实现从descriptor出发反演出原来的结构，这是因为得到descriptor的过程是一个提取特征值的过程，必然存在着信息的损失
  - 即使一个representation的思路是有效的，它产生的descriptor也是合理的，但是descriptor的元素数量不够，则描述得到的仍是incomplete
  - 如何衡量一个representation的有效程度(the degree of faithfulness)
  - 根据上述难点，我们设计了一套reconstruction方案

### Reconstruction Experiment

#### 总体思路

- 选取一个多原子结构作为target
- 以这个target作为original structure，在它原有的结构上做一定的扰动，让它的结构随机发生一定程度的变化
- 接下来我们的目标是从扰动后的结构**reconstruct**原本的结构：
  - 用descriptor分别表示出**target structure** 和 **randomized structure（candidate structure）**
  - 将两个descriptor之间的difference作为目标函数，利用共轭梯度算法求最小值：这是用来比较**两个descriptor的差异性**的
  - 利用前面提到的$d_{ref}$用来比较两个原子结构之间的差异性
  - 具体做法：
    - 调整candidate的结构，用descriptor将其表示
    - 将得到的descriptor和fixed target的descriptor进行比较，用共轭梯度算法指导candidate的结构调整；同时计算每一次调整后的结构和fixed target之间的$d_{ref}$
    - 如果$d_{ref}<10^{-2} \AA^2 $或梯度小于$10^{-8}\AA^2$，则停止算法。
      - 若是因为前者而停，则认为两个结构达到一致，descriptor在这次实验当中成功
      - 若是因为后者而停而未达到前者的要求，则认为经过扰动之后很有可能两个结构完全不同，但descriptor却是相同的，为此必须进一步进行验证：
        - 利用$\Delta(\hat{R})$或是参考文献42中的fingerprints的方法，来具体验证这两个结构是否是不同的
  - **SOAP**也可以利用上述的过程进行比较，不过需要做出一定的调整：
    - 因为**SOAP方法**是直接跳过了descriptor的方法，因此在做梯度下降时，必须换一种方法：利用**similarity**作为目标函数
- 通过不断改变考虑的总原子数n来验证选择的descriptor对于不同原子数的有效性

