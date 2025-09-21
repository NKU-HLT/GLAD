([简体中文](./README_zh.md)|English)

# GLAD: Global-Local Aware Dynamic Mixture-of-Experts for Multi-Talker ASR

[Arxiv](https://arxiv.org/abs/2509.13093) | 
[Paper HTML](https://arxiv.org/html/2509.13093v2)

## Method Overview

GLAD (Global-Local Aware Dynamic Mixture-of-Experts) is a framework designed to address the challenges of multi-talker automatic speech recognition (MTASR). The motivation behind our design is as follows:

- The Mixture-of-Experts (MoE) paradigm is well-suited for MTASR, as it can naturally handle varying numbers of speakers and different levels of speech overlap.
- Speaker information is critical for MTASR, as it enables the model to understand who is speaking, which in turn helps determine what is being said.

Based on these insights, we propose the following framework illustrated below:

<p align="center">
  <img src="assets/glad.png" width="600"/>
</p>
<p align="center"><em>Figure: Overview of the GLAD-SOT architecture, which applies the proposed GLAD to SOT. (a) GLAD-SOT leverages original speech
 features to generate global expert weights for encoder layers. (b) Each MoLE layer combines global and local weights to coordinate LoRA
 experts. (c) The global-local aware dynamic fusion module adaptively fuses global and local weights to guide expert collaboration.</em></p>

Our main contributions are as follows:

-  To our best knowledge, we are the first to apply MoE architectures to end-to-end MTASR, showing significant improvements over conventional methods.
- We introduce GLAD, which dynamically combines speaker-aware global context with fine-grained local acoustic features. This enables experts to focus on target speakers while capturing fine-grained speech patterns in multi-talker scenarios.
- Our ablation studies investigate the role of global acoustic features and their support for speaker-aware expert routing in MTASR. For more details, please refer to our [paper](https://arxiv.org/abs/2509.13093)

## Training Data

**Step 1**: Navigate to the `traindata` directory and run `run.sh` to extract the data. This will generate two folders: `generate` and `traindata`.

**Step 2**:

- The generate folder contains two annotation files:
    - train-960-1mix.jsonl: LibriSpeech-train-960.
    - train-960-2mix.jsonl: Two-talker speech created by mixing audio from two speakers from LibriSpeech-train-960.
- Use the [LibrispeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix) toolkit to generate the mixed audio.
For each sample, the transcript is represented as "text1" (single-talker) or "text1 $ text2" (two-talker), where $ indicates a speaker change.


**Step 3**:

- The `traindata` directory includes:
    - `wav.scp`: An index file processed by ESPnet with speed perturbations (0.9x, 1.0x, 1.1x). This file illustrates the naming convention we used.
    - `wavlist`: A list of audio IDs used as **training data** in our experiments.
- Filter the audio generated in Step 2 using `wavlist` to obtain the training data used in our paper.


## Using GLAD

This project is developed based on the [ESPnet](https://github.com/espnet/espnet) framework. 

GLAD-specific configuration files can be found [here](./espnet/egs2/librispeech/asr1/configs).

**Step 1**:

Replace the `espnet`, `espnet2`, and `egs2` directories in your local ESPnet repository with the corresponding folders from this repo.
Then, update the configuration files (e.g., data paths) according to your setup.


**Step 2**:

Prepare the data and run [`run.sh`](./espnet/egs2/librispeech/asr1/run.sh). 

First, execute the initial stages for data preparation, and then run stages 10 through 13 for training.

**Step 3**:
Use [`run_pi_scoring.sh`](./espnet/egs2/librispeech/asr1/run_pi_scoring.sh) to evaluate the model.

The evaluation code is adapted from [Speaker-Aware-CTC](https://github.com/kjw11/Speaker-Aware-CTC), and we appreciate their open-source contributions.


## Contact
If you have any questions or are interested in collaboration, feel free to contact us via email:

guoyujie02@mail.nankai.edu.cn

## Citation
If you find our work or code helpful, please consider citing our paper and giving this repository a ⭐.

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

## Acknowledgements

This project is built upon the [ESPnet](https://github.com/espnet/espnet) framework.

We would like to thank the following open-source projects, which inspired and supported parts of our implementation:

- [LibrispeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- [Speaker-Aware-CTC](https://github.com/kjw11/Speaker-Aware-CTC)