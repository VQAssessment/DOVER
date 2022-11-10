# DOVER: the Disentangled Objective Video Quality Evaluator

![visitors](https://visitor-badge.laobi.icu/badge?page_id=teowu/DOVER) 

:sparkles: Arxiv Preprint Link: [abs](arxiv.org/abs/2211.04894), [pdf](arxiv.org/pdf/2211.04894).
:sparkles: We will add an appendix about more technical details soon!

The first attempt to disentangle the VQA problem into aesthetic and technical quality evaluations.
Official code for ArXiv Preprint Paper *"Disentangling Aesthetic and Technical Effects for Video Quality Assessment of User Generated Content"*.







## Introduction

### Problem Definition

![Fig](figs/problem_definition.png)

### the proposed DOVER


![Fig](figs/approach.png)

## Install

The repository can be installed via the following commands:
```shell
git clone https://github.com/teowu/DOVER.git \
cd DOVER \
pip install . \ #stop here if you do not need pretrained weights, but why not?
mkdir pretrained_weights \ 
wget https://github.com/teowu/DOVER/releases/download/v0.1.0/DOVER.pth -> pretrained_weight
```

## Judge the Quality of Any Video

### Try on Demos

You can run a single command to judge the quality of the demo videos in comparison with videos in VQA datasets.

```shell
    python evaluate_one_video.py -v ./demo/17734.mp4
```

or 

```shell
    python evaluate_one_video.py -v ./demo/1724.mp4
```

### Evaluate on your customized videos


Or choose any video you like to predict its quality:


```shell
    python evaluate_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```

### Outputs

You should get some outputs as follows. As different datasets have different scales, an absolute video quality score is useless, but the comparison on both **aesthetic** and **techincal quality** between the input video and all videos in specific sets are good indicators for how good the quality of the video is.

In the current version, you can get the analysis of the video's quality as follows (the normalized scores are following `N(0,1)`, so you can expect scores > 0 are related to better quality).


```
Compared with all videos in the LIVE_VQC dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 40% of videos, with normalized score 0.02.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 64% of videos, with normalized score 0.50.
Compared with all videos in the KoNViD-1k dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 72% of videos, with normalized score 0.70.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 91% of videos, with normalized score 1.20.
Compared with all videos in the LSVQ_Test dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 67% of videos, with normalized score 0.52.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 78% of videos, with normalized score 0.84.
Compared with all videos in the LSVQ_1080P dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 50% of videos, with normalized score 0.18.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 54% of videos, with normalized score 0.24.
```


## Data Preparation

We have already converted the labels for every dataset you will need for Blind Video Quality Assessment,
and the download links for the videos are as follows:

:book: LSVQ: [Github](https://github.com/baidut/PatchVQ)

:book: KoNViD-1k: [Official Site](http://database.mmsp-kn.de/konvid-1k-database.html)

:book: LIVE-VQC: [Official Site](http://live.ece.utexas.edu/research/LIVEVQC)

:book: YouTube-UGC: [Official Site](https://media.withyoutube.com)

After downloading, kindly put them under the `../datasets` or anywhere but remember to change the `data_prefix` in the [config file](dover.yml).


## Dataset-wise Default Inference

To test the pre-trained DOVER on multiple datasets, please run the following shell command:

```shell
    python default_infer.py
```

## Visualization

### Divergence Maps

Please follow the instructions in [Generate_Divergence_Maps_and_gMAD.ipynb](Generate_Divergence_Maps_and_gMAD.ipynb) to generate them.
You can also get to visualize the videos (but you need to download the data first).

### WandB Training and Evaluation Curves

You can be monitoring your results on WandB!
Though training codes will only be released upon the paper's acceptance, you may consider to modify the [FAST-VQA's fine-tuning scripts](https://github.com/teowu/FAST-VQA-and-FasterVQA/blob/dev/split_train.py) as we have done to reproduce the results.

Or, just take a look at our training curves that are made public: 

[Official Curves](https://wandb.ai/timothyhwu/DOVER)

and welcome to reproduce them!


## Results

### Score-level Fusion

Directly training on LSVQ and testing on other datasets:

|    | PLCC@LSVQ_1080p | PLCC@LSVQ_test | PLCC@LIVE_VQC | PLCC@KoNViD | MACs | config | model |
| ----  |    ----   |   ---- |  ----   |    ----   | ----  |    ----   |   ---- | 
|  DOVER |  0.830 |  0.889  |   0.855 | 0.883   |  282G  |  [config](dover.yml)  | [github](https://github.com/teowu/DOVER/releases/download/v0.1.0/DOVER.pth) |

### Representation-level Fusion

Transfer learning on smaller datasets:

|       | KoNViD-1k | CVD2014 | LIVE-VQC | YouTube-UGC |
| ----  |    ----   |   ---- |  ----   |    ----   |
| SROCC | 0.906 | 0.894 | 0.858 | 0.880 |
| PLCC  | 0.905 | 0.908 | 0.874 | 0.874 |



## Acknowledgement

Thanks for [Annan Wang](https://github.com/AnnanWangDaniel) for developing the interfaces for subjective studies.
Thanks for every participant of the studies!

## Citation

Should you find this work interesting and would like to cite this, please feel free to add these in your papers!

```bibtex
@article{wu2022disentanglevqa,
  title={Disentangling Aesthetic and Technical Effects for Video Quality Assessment of User Generated Content},
  author={Wu, Haoning and Liao, Liang and Chen, Chaofeng and Hou, Jingwen and Wang, Annan and Sun, Wenxiu and Yan, Qiong and Lin, Weisi},
  year={2022}
}
```