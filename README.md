# 成为 FDE(Forward Deployed Engineer,前线部署工程师):系统教程

本仓库是一套教科书式的系统课程,而非单篇文章。每章包含:学习目标、核心概念与定义、原理、实例、图示(Mermaid)、案例、练习(含参考答案或评分标准)、小结、关键术语与延伸阅读(仅收录已核实来源)。

## 适合谁
- 有 1 年以上编程经验、希望转向"工程 + 客户"复合岗位的软件工程师
- 想补齐工程能力的解决方案工程师(SE)、技术咨询顾问、数据分析师
- 计算机相关专业的高年级学生

## 如何使用
1. 先读 `00-overview` 建立对岗位的准确认知;
2. 再读 [预备章 Linux 基础](01-foundations/00-linux-basics.md)（不占章号），然后进入第 1、2 章;
3. 按 `ROADMAP.md` 的 12 周或 24 周计划推进;
4. 每章练习务必动手;`04-projects` 的作品是求职的核心证据;
5. 面试前两周集中练 `05-interview`;
6. 第 3 章在目录 `02-ai-llm`(目录编号不是章号)。12 周路线读 3.1–3.2。只有岗位涉及模型训练或推理性能时,再读 3.3–3.9;那几节是选修,不是全体 FDE 的必修。

## 目录
| 部分 | 章节 | 子章节 |
|---|---|---|
| 00 概览 | [第 0 章 FDE 岗位全景](00-overview/README.md) | [0.1 定义](00-overview/README.md#01-定义)<br>[0.2 起源](00-overview/README.md#02-起源)<br>[0.3 与相邻岗位对比](00-overview/README.md#03-与相邻岗位对比)<br>[0.4 日常工作](00-overview/README.md#04-日常工作典型一周)<br>[0.5 取舍与适配度](00-overview/README.md#05-取舍与适配度)<br>[0.6 职业路径](00-overview/README.md#06-职业路径)<br>[0.7 通识与专项](00-overview/README.md#07-通识与专项) |
| 01 工程基础 | [预备章 Linux 基础](01-foundations/00-linux-basics.md) | [L.1 架构](01-foundations/00-linux-basics.md#l1-linux-架构内核用户态与发行版)<br>[L.2 文件系统](01-foundations/00-linux-basics.md#l2-文件系统层次inode-与一切皆文件)<br>[L.3 Shell 与命令行](01-foundations/00-linux-basics.md#l3-shell-与核心命令行)<br>[L.4 用户、权限与远程访问](01-foundations/00-linux-basics.md#l4-用户权限与远程访问)<br>[L.5 进程与 systemd](01-foundations/00-linux-basics.md#l5-进程与-systemd)<br>[L.6 软件包、环境与共享库](01-foundations/00-linux-basics.md#l6-软件包环境变量与共享库)<br>[L.7 网络基础](01-foundations/00-linux-basics.md#l7-网络基础)<br>[L.8 资源排查与性能](01-foundations/00-linux-basics.md#l8-资源排查与性能)<br>[L.9 容器与 GPU 服务器](01-foundations/00-linux-basics.md#l9-容器的-linux-基础与-gpu-服务器)<br>[L.10 陌生服务器前 10 分钟](01-foundations/00-linux-basics.md#l10-到达陌生客户服务器的前-10-分钟) |
| | [第 1 章 系统集成与数据工程](01-foundations/01-integration-and-data.md) | [1.1 核心概念](01-foundations/01-integration-and-data.md#11-核心概念)<br>[1.2 原理:为什么集成很难](01-foundations/01-integration-and-data.md#12-原理为什么集成很难)<br>[1.3 API 集成要点](01-foundations/01-integration-and-data.md#13-api-集成要点)<br>[1.4 数据探查与 SQL](01-foundations/01-integration-and-data.md#14-数据探查与-sql)<br>[1.5 可观测的数据管线](01-foundations/01-integration-and-data.md#15-可观测的数据管线) |
| | [第 2 章 部署、运维与安全](01-foundations/02-deployment-and-security.md) | [2.1 核心概念](01-foundations/02-deployment-and-security.md#21-核心概念)<br>[2.2 部署形态](01-foundations/02-deployment-and-security.md#22-部署形态)<br>[2.3 实例:最小可部署单元](01-foundations/02-deployment-and-security.md#23-实例最小可部署单元)<br>[2.4 运维](01-foundations/02-deployment-and-security.md#24-运维)<br>[2.5 安全](01-foundations/02-deployment-and-security.md#25-安全) |
| 02 AI/LLM | [第 3 章 语言模型应用与系统](02-ai-llm/README.md) | [3.1 业务与选型](02-ai-llm/01-business-and-model-choice.md)<br>[3.2 应用:RAG、Agent、评测](02-ai-llm/02-applications-rag-agent-eval.md)<br>[3.3 预训练与适应](02-ai-llm/03-pretraining-and-adaptation.md)（选修）<br>[3.4 微调与对齐](02-ai-llm/04-finetuning-and-alignment.md)（选修）<br>[3.5 神经网络与 Transformer](02-ai-llm/05-neural-nets-and-transformers.md)（选修）<br>[3.6 PyTorch](02-ai-llm/06-pytorch-engineering.md)（选修）<br>[3.7 训练与推理系统](02-ai-llm/07-training-and-inference-systems.md)（选修）<br>[3.8 CUDA](02-ai-llm/08-cuda-and-gpu-kernels.md)（选修）<br>[3.9 Linux 存储栈](02-ai-llm/09-linux-io-and-storage-stack.md)（选修） |
| 03 客户能力 | [第 4 章 发现、界定与干系人管理](03-customer-skills/README.md) | [4.1 核心概念](03-customer-skills/README.md#41-核心概念)<br>[4.2 发现方法](03-customer-skills/README.md#42-发现方法)<br>[4.3 从目标到范围](03-customer-skills/README.md#43-从目标到范围)<br>[4.4 干系人管理](03-customer-skills/README.md#44-干系人管理)<br>[4.5 演示与写作](03-customer-skills/README.md#45-演示与写作)<br>[4.6 处理模糊与变更](03-customer-skills/README.md#46-处理模糊与变更) |
| 04 实战项目 | [第 5 章 作品集项目](04-projects/README.md) | [P1 多源数据整合与看板](04-projects/README.md#p1-多源数据整合与看板)<br>[P2 企业知识库 RAG](04-projects/README.md#p2-企业知识库-rag)<br>[P3 文档抽取工作流](04-projects/README.md#p3-文档抽取工作流)<br>[P4 工具调用 Agent](04-projects/README.md#p4-工具调用-agent)<br>[P5 私有化部署包](04-projects/README.md#p5-私有化部署包)<br>[P6 制度助手（选修）](04-projects/README.md#p6选修制度助手的容量与发布计划) |
| 05 面试 | [第 6 章 面试准备](05-interview/README.md) | [6.1 常见环节](05-interview/README.md#61-常见环节具体以各公司为准)<br>[6.2 问题分解面试](05-interview/README.md#62-问题分解面试)<br>[6.3 编码](05-interview/README.md#63-编码)<br>[6.4 行为与客户场景](05-interview/README.md#64-行为与客户场景)<br>[6.5 专项系统问题](05-interview/README.md#65-专项岗位可能追问的系统问题)<br>[6.6 两周冲刺清单](05-interview/README.md#66-两周冲刺清单) |
| 06 职业 | [第 7 章 求职与职业发展](06-career/README.md) | [7.1 目标公司](06-career/README.md#71-目标公司招聘页已于-2026-11-访问核实具体岗位请以页面为准)<br>[7.2 简历](06-career/README.md#72-简历)<br>[7.3 作品集](06-career/README.md#73-作品集)<br>[7.4 谈判](06-career/README.md#74-谈判)<br>[7.5 成长](06-career/README.md#75-成长) |
| 资源 | [精选资源](resources/README.md) | [书籍](resources/README.md#书籍)<br>[文章](resources/README.md#文章)<br>[课程与参考](resources/README.md#课程与参考)<br>[专项:GPU、训练与推理](resources/README.md#专项gpu训练与推理) |

## 学习计划概要
- **12 周(全职强化)**:第 1–2 周概览、Linux 基础与集成;3–4 周部署安全;5–7 周 LLM 系统;8 周客户能力;9–11 周项目;12 周面试。
- **24 周(在职)**:每阶段时长翻倍,项目阶段并行进行客户能力练习。
详见 [ROADMAP.md](ROADMAP.md)。第 5–7 周读第 3.1–3.2 节;3.3–3.9 是同一章里的选修,不替换上面的 12 周。

## 说明
本教程不提供薪资数字或未经核实的统计;岗位描述以各公司官网招聘页为准。欢迎按 [CONTRIBUTING.md](CONTRIBUTING.md) 补充。
