---
title: 'VALSE 2021'
date: 2021-10-8
permalink: /posts/valse/
tags:
  - meeting record

---

# VALSE 2021
## day1
### keynote session

- 徐宗本 如何突破机器学习的先验假设？ On presuppoditions of machine learining: a best-fitting theory
    - outline 
    - machine learining with hypotheses
        - the independence hypotheses on loss function
        - the large capacity hypothesis on hypothetical space
        - the completeness hypothesis on training data
            - curriculum learning 课程学习，从易到难 [TPAMI 2021](https://zhuanlan.zhihu.com/p/362351969)
            - self-paced learning 
        - the prior determination on regularizer
        - the euclidean space hypothesis on analysis

### 特邀报告

- 唐华锦 多神经环路协同的类脑学习与计算架构探索
    - snn很强，亿级神经元类脑计算机，然鹅不懂

- 黄高 动态神经网络研究进展与展望
    - why architecture matters?
        - representation power
        - opt
        - generalization
        - efficiency

    - 静态网络 vs 动态网络
        - 通过样本的变化改变网络的参数
        - why we need dynamic
            - bigger is better 人的视觉系统也是动态的
    
    - dynamic nn
        - sample-wise dynamic nn
            - dynamic depth：early exiting
            - dynamic vision transformer
            - skip branches：Mixture of experts（MOE）
            - dynamic routing in supernets
            - parameter adjustment
            - kernel shape adaption
        - spatial-wise dynamic nn
            - pixel-level dynamic nn
        - temporal-wise dynamic nn
            - 时序处理
    
    - advantage & challenges of dynn

### 年度进展

- 白翔 场景文字理解的经典工作回顾和未来展望

- 程健 神经网络模型压缩领域年度进展概述
    - 无预训练剪枝
    - 混合精度量化
        - Auto Q
    - 训练阶段加速
    - 注意力模块压缩
    - 软硬件协同设计

- 周晓巍 三维人体重建年度进展
    - 问题定义
        - 三维骨架 >> 精细化模型
    - 热点方向
        - 运动捕捉
            - 考虑物理约束的运动捕捉
            - 利用特定场景中的额外约束 （无/有镜面约束）
        - 表面重建
            - PiFu：基于神经表示的单目几何重建 surface / texture
            - PiFuHD：coarse to fine
            - PAMIR
        - 神经渲染
            - 利用神经渲染提升视图合成的真实感
            - 视图合成的里程碑式工作：神经辐射场 NeRF
            - neural body：动态人体视图合成
            - 可形变的神经辐射场
        - 可驱动建模
            - 从图像重建可驱动的三维人体模型
            - 可驱动的神经辐射场：LBS+NERF
        
        - 未来展望
            - 多人或者人与环境交互
            - 高效可泛化的神经建模与渲染
            - 可控可编辑的神经渲染
                - 姿态迁移等
            - 多模态运动与表情驱动
            - 重光照


- 邓伟洪 视觉情感计算
    - limitation of emotionnal models
    - RAF-AU Dataset


- 张姗姗 行人检测与重识别年度进展综述
    - 行人检测 Detection
        - occlusion handling
            - NMS 
            - Beta RCNN
        - generalization
            - FAM
    - 行人重识别 Re-ID
        - image-based vs video-based
        - supervised / unsupervised
        - cross-domain
        - cloth-changing
            - cloth consistent vs cloth-changing re-ID
    - person search
        - joint detection & Re-ID

- 巫英才 体育比赛视频的智能处理和可视分析
    - 挑战
        - 遮挡模糊
        - 数据需要专家标注
    - 交互式数据推理框架
        - 多层次数据获取（对象层次、事件层次 etc）
    - 数据分析评估
        - 传统：视频驱动、数据驱动
        - 解决方案：反绎学习

- 韩军伟 AI赋能高分遥感目标检测与识别：挑战、对策及展望
    - 研究背景
    - 挑战
        1. 实现旋转不变特征提取
        2. 有向目标检测，传统依赖水平候选框
        3. 弱监督目标检测
        4. 小样本目标鲁棒检测
        5. 定位型号目标局部判别特征
    - 对策
        1. 旋转不变层 正则
        2. Oriented RCNN & AOPG：Anchor-free
        3. Weekly Supervised learning based on Bayesian Framework & RINet
        4. SPNet: Siamese-Prototype network
        5. P-CNN: Part-Based Convolutional Neual networks 部件定位网络

- 俞扬 环境模型学习——让强化学习走出游戏
    - intro
        - math model： markov decision process (MDP)
        - computer games = known transition function
        - hand-crafted environment?
        - challenge of RL from data 
        - RL from data : offline-RL
    - learn an environmental model
        - historical action- response data
        - supervised model fitting: point matching
        - simulation lemma 
        - error propagation analysis
            - point matching
        - new theory: distribution matching 

- 苏航 深度学习对抗安全方法与评测
    - black-box attacks
        - to solve the problem : 
    - a limitation of the ML Framework
    - adversarial learning
        - stochasic noise
        - xxx noise
    - boosting AT with hypersphere embedding
        - hypersphere embedding (HE) mechanism
        - adversarial training (AT)
    - who goes first? attacker or defender?
    - why adversarial examples?
    - gradient leaking hypothesis
    - label smoothing
    - privacy protection via adversarial attacks

## day2
### keynote session
- 刘成林 模式识别：从分类到理解
    - 模式识别定义与发展历史
        - 如何使机器模拟人的感知功能
        - 任务
            - 物体分类
            - 物体检测
            - 物体分割
            - 图像描述（captioning）
        - 挑战
            - 标记数据的代价
            - 人的真正学习方式
            - end-to-end模型真的理解了模式吗
    - 从分类到模式理解
        - 方法
            - statistical
            - syntactic，structural
            - nn，connectionist
        - 早期的工作试图统一分类和理解
        - 结构方法并没有广泛接受
        - 例子
            - 手写汉字识别
            - 图像分本匹配
            - 混合神经网络
            - 可解释性
    - 趋势分析
        - 当前状况
            - 分类 较为完善了
            - 模式理解 有待讨论
        - 最新趋势
            - 统计（神经）+ 结构模型
            - 可解释性神经网络
            - 类人记忆模型和学习方法
        - 早期思想方法值得借鉴
            - 类人模式识别：模板匹配，特征匹配，原型分类器，多层次理解
            - 结构表示匹配、句法方法、知识驱动方法
    - 未来研究方向
        - 结构表示模型
            - 神经网络+结构
            - 跨模态
        - 结构模型学习
            - 图匹配度量学习
            - 半监督/弱监督学习，课程学习
            - 开放环境增量学习，自适应，小样本，领域泛化
            - 跨模态学习（如文本监督），交互学习
            - 用好自监督学习：预训练模型可大为简化结构模型学习
        - 语义理解应用
            - 感知+认知+决策，利用背景知识和上下文


### 年度进展 APR

- 杨易 视频分析领域年度进展概述 video analysis
    - new architectures
        - cnn vs transformers
            - ViViT iccv 2021
    - new dataset 
        - large scale ：scale up
        - eg：webViD
    - efficiency
        - replace expensive modules
        - decouple expensive operations
        - search for light weight architectures
        - fast inter-frame correlation operation
        - Sparse sampling
    - Better semantic alignment

- 崔鹏 分布外泛化(Out-Of-Distribution generalization)
    - problem definition
        - I.I.D. learning vs Out-Of-Distribution (即train和test有关联)
    - OOD methods
        - Unsupervised representation learning
        - Supervised model learning
        - Optimization
    - Causal Representation

- 刘偲 视觉与语言年度进展综述
    - pretraining
        - image & text: single-Tower model
        - bert系列
    - vision and language navigation
        - knowledge reasoning 
        - structured scene memory
    - Visual grounding
        - progressive comprehension
    - VQA
        - scanpaths
        - skill-concept
    - visual captioning generation
        - general visual captioning        
            - structured visual coding
            - obtain additional information
        - dense video captioning
        - text-aware image captioning
    - text-guided image synthesis
        - contrasive learning
        - DALL-E
        - cross modal cyclic 
    - future directions
        - general vision language
        - specific domain

- 谭明奎 对抗视觉生成与合成年度进展概述
    - 视觉生成
        - 老电影着色
        - 游戏场景生成
        - AR/VR
        - 虚拟数字人
    - 判别模型 vs 生成模型
        - 判别：CNN SVM
        - 生成：VAE GAN
    - GAN
        - 传统的缺点
            - 清晰度
            - 多样性
            - 可控性
        - 视觉生成典型任务
            - 图像生成
            - 视频生成

- 张磊 迁移学习年度进展
    - non iid
    - 十个问题 10 branches in today' S transfer learning
        1. Local alignment with conditional shift (条件偏移问题)
        2. Pseudo-labeling for self- training (伪标注质量问题)
        3. Universal adaptation with category shift (通用迁移问题)
        4. Multi- source domain adaptation (多源迁移问题)
        5. Source-free domain adaptation due to privacy (数据隐私问题)
        6. Balancing alignment and discrimination (任务偏见问题)
        7. Replacement of entropy minimization (预测偏见问题)
        8. Domain generalization (未知域泛化、0OD外推问题)
        9. Transferable robustness and trustworthiness (迁移鲁棒和可信问题)
        10. Transfer learning + X (迁移学习的应用)

### workshop 具身智能视觉
- 卢策吾 机器人抓取云端竞赛OCRTOC 与AnyGrasp（基于GraspNet）
    - Open cloud robot platform
        - IROS 2020
        - OCRTOC(Open Robots for table organization challenge) 2021：labelled the real world data
        - grasping & manipulation is hard
            - suction cup is often preffered
            - grasping unknown objects is difficult
            - robots are far behind humans

    - Embodied
        - accurate and complete grasp pose
        - semi-automatic 
    - robotflow 
        - 结构
            - 人类指导层
            - 具身感知(Perception)、具身想象(Imagination)、具身执行(Execution)
            - 具身智能系统应用
        - 基于python开发
        - RF Vision
            - 2D\3D目标检测
            - 刚体、关节体柔性体参数估计
            - 手物操作重建
            - 视触多模态感知
        - RF Universe
            - 对概念的全面仿真
        - RF Planner
            - 任务和动作规划库
        - RF Controller
            - 执行

- 黄刚 智能驾驶虚拟仿真测试与训练平台
    - 研发背景
        - 需要适应各种路况
        - 全面系统测试
    - 需要虚拟仿真
    - 平台架构
        - 场景建模
        - 仿真运行（感知、决策、控制）
        - 数据分析
    - 核心功能
        - 智能案例生成
        - 存档回放服务
    - 关键技术
        - 数字孪生静态要素建模
            - 大规模场景 ，去遮挡
        - 数字孪生动态要素建模
            - 动态的人、车
        - 数字孪生环境要素建模
            - 光照、雾天
        - 高保真传感器仿真
            - 可见光、红外、激光雷达、毫米波
        - 交通参与者仿真
            - 驾驶员的个人因素等
        - 场景仿真
            - 信号灯、路边设施等
        - 标准案例库
        - 对抗训练
            - 强化学习

- 杨睿刚 Autonomous Driving Trucks: Challenges and Opportunities.
    - 卡车司机现状
    - Challenges
        - Truck vs taxi: Break, Lane Ch, Weight, Gas up!
    - Truck-Specific AD Algorithm
        - Long Range 3D Sencing: Lidar & Rader
            - Limitation in Current Approaches: 现有 Depth from mono camera
                - Method: Extend LiDAR via unsupervised Learning: Depth map， 近处的车辆， 2 fps/s
                - Foreground vehicle fitting: 远处的车辆检测
        - Lidar scene parsing
    - 模型的训练策略：以车养车，逐渐进行智能level的提高

- 王鹤 类别级六维位姿估计与追踪
    - Definition 3个旋转以及平动自由度
    - 6 pose estimation
    - instance-level 6D pose estimation
    - normalized object coordinate 归一化物体坐标
        - scale normalization：uniformly normalize the scales
        - NOCS
    - ANCSH: articulation-aware normalized coordinate space hierarchy
        - NAOCS normalized articulated object coordinate space
        - NPCS normalized part coordinate space
    - online pose tracking

- 樊庆楠 如何解决动态室内场景的相机定位问题？
    - outlier-aware neural tree
        - hierarchical space partition
        - deep routing function
        - outlier rejection
    - active visual localization
        - markov localization
            - greedy action
            - learned action by reinforcement learning 
        - limitation
            - complete map construction
            - localization accuracy
            - huge state space
            - assumption of perfect agent movement for Bayesian filting
    

- spotlight
    - Perceptual Question Answering CVPR2021 BUPT
    - Towards Distraction-Robust Active Visual Tracking ICML2021 PKU


## day3
### workshop 三维视觉技术
- 郭裕兰 点云深度学习新探索：更大、更少、更通用
    - 大规模场景下的点云语义分割 
        - PointNet和PointNet++为何不能处理大规模点云
        - RandLA-Net [TPAMI2021](https://github.com/QingyongHu/RandLA-Net) 建筑物级别 -> 城市级别
        - 鱼和熊掌如何得兼？
            - 高效的点云降采样：降低显存和计算消耗
            - 有效的局部特征聚合：对几何模式进行有效编码

        - 如何有效地保留局部几何信息

    - 少量标注数据下的点云语义分割
        - 数据量急剧增加
        - 数据标注成本提升
        - 点云语义分割是否真的需要密集标注
        - 如何充分利用极少量的标注点实现网络参数的更新？

    - 泛化性强的点云局部特征描述
        - [SpinNet](https://github.com/QingyongHu/SpinNet): Learning a General Surface Descriptor for 3D Point Cloud Registration (CVPR 2021) 
        - 点云局部特征描述子在配准中的作用
        - 不同传感器获得的点云差异大
        - 现有方法在不同数据集上的泛化能力弱
        - 局部特征的旋转不变性
            - 法向量对齐：消除两个角度自由度
            - 3D柱状表示：消除绕法向量旋转的自由度
        - 特征的泛化能力
            - 旋转不变带来的场景适应性
            - 柱状卷积获得的感受野增加
        - 泛化能力如何
            - 训练集：3DMatch -> 测试集：ETH
    - 未来工作
        - 点云数据规模大：无监督/弱监督学习
        - 点云数据差异大：对不同任务/不同传感器数据的强泛化性

- 周晓巍 Learning-based 3D Reconstruction and Localization: What should be learned?
    - Two fundamental problems in 3D vision
        - Reconstruction and Localization
    - Structure from Motion
    - Learning to estimate pose
        - Using neural network to directly predict camera pose from image?
            - [PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://github.com/PoseNet-Mobile-Robot/Mobile-Robotics)
            - [Unsupervised Learning of Depth and Ego-Motion from Video](https://github.com/tinghuiz/SfMLearner)
    - learning to find correspondence 
        - MatchNet
        - LIFT
        - Toward Geometric deep SLAM
        - SuperPoint
    - learning-based local features
        - SIFT vs learning
        - learning from description using camera pose supervision ECCV 2010
        - eg. SenseMARS AR导航
    - Remaining challenge
        - detector repeatability
        - descripor distinctiveness
            - local descripter do not encode postional and global information
    - LoFTR: Detector-Free Local Feature Matching with Transformers CVPR 2021
        - detector-free matching
            - coarse-level matching 
            - fine-level matching 
        - Local feature transformer 
        - architecture
            - Local Feature CNN
            - Coarse-Level Local Feature Transform
            - Matching Module
            - Coarse-to-Fine Module 
        - end-to-end learnable feature matching system 
        - Qualitative evalution
            - LoFTR
            - SuperGlue
    - Dense reconstruction
        - depth sensor vs monocular camera
        - current reconstruction systems
            - Depth estimation for keyframe -> depth fusion
            - multi-view stereo (MVS)
            - TSDF Fusion
    - Learning depth estimation
        - MVSNet
        - DeepTAM: deep tracking and Mapping
        - DeepV2D
        - BA-Net
        - Neural RGB-D sensing
    - Do we really need depth as intermediate representation?
        - Depth-based systems
        - Our idea
    - NeuralRocon: 
        - TSDF Fusion
        - End-to-end learnable volumetric reconstruction system
    - Conclusion and discussion
        - learning-based 3D reconstruction and localization
            - Determine what's really necessary to be learned 
            - Design end-to-end learnable systems with differentiable modules
        - Future 3D vision systems
            - Learnable and intelligent (prior/semantics/context, fast initialization)
            - Self-optimizable from observations (e.g. with differentiable rendering)
            - Hardcode well-established geometry theorems (e.g. multi-view geometry)
    - [github](https://github.com/zju3dv)

- 弋力 对比多模态融合 Contrastive Multimodal Fusion
    - Multi-modal Signals Disambiguate Each Other
    - Human Form Concepts While Fusing Multi-modal Signals 
        - Joint Respresenting (Multi-modal space)
        - Unsupervised multimodal representation learning 
        - many application of 3D+X
            - 3D + RGB image 
            - 3D + language [Referlt3D](https://github.com/referit3d/referit3d)
            - 3D + touch [Active 3D Shape Reconstruction from Vision and Touch - CVPR2021](https://arxiv.org/abs/2107.09584)
    - A popular approach: Contrastive learning
        - Contrastive learning
            - postive example 
            - anchor example
            - neative example
        - Unimodal Understanding
        - Multi-modal Understanding
            - CMC
            - MMV FAV
        - Explorations
            - P4Contrast
            - TupleInfoNCE
        - Motivation
            - Unimodal Performance
            - Multimodal Performance
        - Task: Unsupervised RGB-D Representation Learning 
            - without human supervision
            - semantic segmentation
            - other downstream tasks
            - 3D Object Detection
        - PointContrast Revisited
        - Connection with Mutual Information
    - ICCV2021: Contrasive Multimodal Fusion with TupleInfoNCE [paper](https://arxiv.org/abs/2107.02575)
        - TupleInfoNCE Overview
        - Future works
        - Problem to be solved
        - Network architecture: DeRenderNet
            - Data & preprocessing
        - Network architecture: SOLID-Net
        - Inverse rendering from intrinsics
            - Faces
            - Scenes
            - Lights

- 施柏鑫 基于本征属性的逆渲染
    - HyFRIS-Net: Hybrid Face Reflectance, Illumination, and Shape modeling Network
    - problem to be solved 
    - image formation model 
        - $I=A*S+\epsilon$

- 吴毅红 融入学习的SLAM
    - “计算机视觉的根本问题是一个鲁棒性问题”
    - 图像底层特征提取
        - 退化 -> 计算结果不好
        - 非退化 -> 计算结果受底层特征质量影响
    - 深度学习优势
        - 具有强大的特征表达能力
        - 数据驱动，应对退化
        - 高层语义，或用来约束底层特征提取
        - **深度学习融入到传统几何的框架下**
    - 解决思路
        - 深度学习：整幅全局，局部描述子
            - 解决速度较慢的问题：随机森林，二值
            - 解决精度下降的问题：contrain局部 浮点+二值 浮点与二值混合
            - 解决光照的问题：融入小波的全局CNN低光增强
        - 整幅检索与直接2D-3D的随机森林匹配，并行，避免二者不足，而优势兼具
            - 对视角变化、光照鲁棒
        - ref
            - TIP2016
            - PR2021
            - ACM MM2021
    - Loss函数设计
        - 三元损失
        - 二阶相似性
        - 加权的Hamming距离
        - **实值与二值同时训练**
    - 未来工作
        - 将传统几何与深度学习融合：鲁棒性、精度、速度
            - 复杂动态场景下的SLAM
            - 大场景实时在线稠密SLAM
            - 多传感器深度融合的SLAM
            - 场景的综合理解
        
- spotlight         
    - Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling [github](https://github.com/bertjiazheng/Structured3D)


## workshop 视觉与语言

- 金琴 Visual Content Captioning and Visually Assisted Language Translation
    - vision and language
        - description/narration
            - multi-lingual mulyi-modal learning 悟道2.0
        - text-aware image captioning
            - TextCaps
            - VizWiz-Captions
        - question-controlled text-aware image captioning
            - automatic dataset construction
        - multimodal machine translation
            - neural machine trasnlation NMT
            - multimodel machine translation MMT
            - product-oriented machine translation PMT
    - Trustable/Explainable Multimodal Learning: from system interpretability/Generalizability perspective
    - Conclusion
        - vision-language for better human-computer collaboration
            - Fine grained knowledge acquisition via testbeds, algorithms and systems
            - explainable/trustable learning for cross-modal learning
        - future work 
            - large-scale fine-grained cross-media knowledge base construction
            - neural symbolic for cross-media analytics
            - bio-inspired mechanisms for cross-media learning
        

- 王树徽 细粒度图文知识获取与可信推理
    - data driven solutions
        - big data + big model -> better 
        - agnostics 
            - AI accidents
        - human-machine interaction
        - reseach challenge
            - lack of fine-grained knowledge
            - blackbox AI systems
        - multi-modal entity linking
        - algorithm-video actor and action segmentation from a sentence
        - algorithm-adaptive reconstruction network
            - weakly supervised REG
        - system-cross-media knowledge engineering
            - human-computer collaboration
                - machine side
                - human side 
                - human-machine
    - explainable AI
        - transparent design
            - Bayesian transparent VQA
            - trustable/debias learning - robust VQA against language bias
            - human-like learning 
        - post-hoc explanation
    - conclusion

- 刘偲 跨模态分析
    - 人-物关系检测(Human Object Interaction Detection & Segmentation)
        - 提出
            - 传统：现实场景存在较多干扰实例，难以高效精准定位有交互的人-物
            - 思路：提出关系点引导的并行定位匹配交互算法
        - PPDM：单阶段的视觉关系定位模型
        - 自适应匹配的人-物交互定位
        - 级联解耦网络
        - 视觉关系分割
    - 图像视频指代分割(Image & Video Referring Segmentation)
        - 图像指代分割
            - 动机和思路：利用不同类型单词的语义信息分阶段地找到被指代的物体，细化多模态特征交互过程
            - CMPC：渐进式跨模态融合指代分割模型
            - 实体感知阶段
            - 关系推理阶段
            - 可视化
        - 视频指代分割
    - 远程视觉指代定位(Remote Embodied Visual Referring Expression, REVERIE)
        - 基于Transformer的网络结果
            - Encoder：文本信息编码
            - Decoder：序列决策
        - ROAA（Room and Object Aware Attention）机制
            - 在文本、视觉信息中提取房间/物体相关的语义信息
        - embodied AI
        - R2R 视觉语言导航
    - 语言指导的图像编辑(Texy Guided Image Editing)
        - 基于语言指导的图像修饰
            - 现有工作：描述全图细节以改变图片属性
            - 提出工作：描述编辑动作以编辑图片
                - EDNet（编辑描述网络）：输入编辑前图像和编辑后图像，生成描述该图像变换的编辑嵌入。
                - 循环机制：通过EDNet和生成器可构成循环机制，使用swapping augmentation 和 XX augmentation，可提升EDNet的XX。
            - 跨模态的循环机制：缓解配对数据不足和不平衡的问题

## workshop Transformer and Attention for Vision
- 任文琦 注意力和Transformer在图像复原中的应用
    - graph neural network
        - SRGAT：Single Image Super-Resolution With Attention Network (TIP2021)
        - image patches tend to recur abundantly in natural images
    - transformer fusion network
        - siamese-Mixer 参考了mlp-mixer
    - image processing transformer (IPT)
        - pre-trained image processing transformer (CVPR 2021)
        - Network Architecture
            - multi-head
            - a shared transformer body
            - multi-tail

    - swin IR 
        - swinIR: image restoration using swin transformer (ICCVW 2021)
        - network architecture
            - shallow feature extraction
            - deep feature extraction
                - residual Swin Transformer blocks (RSTB)
                - image reconstruction

- spotlight
    - NLH: A Blind Pixel-level Non-local Method for Real-world Image Denoising. TIP-2020 [github](https://github.com/njusthyk1972/NLH)
    - OCRNet: Object Contextual Representations for Semantic Segmentation - an Attention View ECCV2020 [paper](https://arxiv.org/pdf/1909.11065.pdf)