(简体中文|[English](./README.md))

# GLAD: Global-Local Aware Dynamic Mixture-of-Experts for Multi-Talker ASR

[Arxiv](https://arxiv.org/abs/2509.13093) | 
[Paper HTML](https://arxiv.org/html/2509.13093v2)

## 方法框架

GLAD（Global-Local Aware Dynamic Mixture-of-Experts）是一个解决多说话人语音识别（Multi-talker ASR， MTASR）的方法框架。我们的动机是：

- MoE的思想非常契合MTASR，可以用来解决不同数量的说话人和不同难度的重叠语音。
- 说话人的信息对MTASR非常重要，能够帮助模型知道“谁在说话”，进而知道“说了什么”。

因此，我们提出了下图所示的方法框架：

<p align="center">
  <img src="assets/glad.png" width="600"/>
</p>
<p align="center"><em>Figure: GLAD-SOT的整体框架，GLAD-SOT是将GLAD应用到SOT中。 (a) GLAD-SOT 使用最原始的语音生成全局专家权重，并广播给encoder中的每个线形层。 (b) 每个MolE将全局权重和局部权重进行融合生成专家最后的权重，去知道对应的LoRA专家。 (c) 全局-局部动态感知模块自适应地融合全局和局部地权重来指导专家们协助工作。</em></p>

我们的贡献是：

- 据我们所知，这是首次将 MoE 架构应用于 MTASR 中，并取得了性能的提升。
- 我们提出了GLAD方法，该方法动态融合了说话人感知的全局上下文信息与细粒度的局部声学特征，使专家模块能够在多说话人场景中聚焦于目标说话人，同时捕捉精细的语音模式。
- 我们通过消融实验深入分析了全局声学特征的作用，以及其在 MTASR 中对说话人感知专家路由的支持。详见我们的[论文](https://arxiv.org/abs/2509.13093)

## 训练数据

步骤1：进入`traindata`目录，运行`run.sh`解压数据。解压后将生成`generate`和`traindata`两个目录。

步骤2：

- `generate`目录下包含两个标注文件：
    - `train-960-1mix.jsonl`：LibriSpeech-train-960
    - `train-960-2mix.jsonl`：通过混合两个说话人音频构建的双说话人数据。
- 使用[LibrispeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)工具生成混合音频。每条语音对应的文本以"text1"（单说话人）和"text1 $ text2"（双说话人）的形式表示，其中"$"表示说话人转换。

步骤3：

- `traindata`目录下包含
    - `wav.scp`：经espnet处理（经过0.9，1.0，1.1变速）生成的索引文件。这里我们给出`wav.scp`是明确我们对文件的命名格式。
    - `wavlist`：本次实验**训练数据**所使用的音频ID列表。
- 根据`wavlist`对步骤2中生成的音频进行过滤，得到本论文实验所需的训练数据。


## 使用GLAD

本项目基于[ESPnet](https://github.com/espnet/espnet)框架进行开发。GLAD详细的配置文件在[这里](./espnet/egs2/librispeech/asr1/configs)。

步骤1：将本仓库中espnet目录下的`egs2`，`espnet`，`espnet2`目录替换至官方[ESPnet](https://github.com/espnet/espnet)仓库对应目录中，并根据实际情况修改配置（如数据路径等）。


步骤2：准备好数据，运行[`run.sh`](./espnet/egs2/librispeech/asr1/run.sh)。请先运行run.sh脚本的数据处理阶段的stage，之后再运行stage10~stage13。

步骤3：利用[`run_pi_scoring.sh`](./espnet/egs2/librispeech/asr1/run_pi_scoring.sh)进行模型评估。评估代码参考自[Speaker-Aware-CTC](https://github.com/kjw11/Speaker-Aware-CTC)，感谢其开源支持。

## 联系我们
如有问题或合作意向，欢迎通过邮箱与我们联系：

guoyujie02@mail.nankai.edu.cn

## 引用
如果我们的工作或代码对您有所帮助，请考虑引用本项目对应论文，并对本项目并给予⭐支持。

```
@misc{guo2025gladgloballocalawaredynamic,
      title={GLAD: Global-Local Aware Dynamic Mixture-of-Experts for Multi-Talker ASR}, 
      author={Yujie Guo and Jiaming Zhou and Yuhang Jia and Shiwan Zhao and Yong Qin},
      year={2025},
      eprint={2509.13093},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.13093}, 
}
```

## 致谢

本仓库是基于[ESPnet](https://github.com/espnet/espnet)框架。

部分实现参考并借鉴了以下开源项目，特此致谢：

- [LibrispeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- [Speaker-Aware-CTC](https://github.com/kjw11/Speaker-Aware-CTC)