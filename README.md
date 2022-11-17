
## DOVER

Official Codes, Demos, Models for the Disentangled Objective Video Quality Evaluator (DOVER)


![visitors](https://visitor-badge.laobi.icu/badge?page_id=teowu/DOVER)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/teowu/DOVER)
<a href="https://colab.research.google.com/github/taskswithcode/DOVER/blob/master/TWCDOVER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 

:sparkles: Our appendix for ***more visualized results*** are added in our updated version of Arxiv Preprint (20 pages, 19 figures)! [pdf](arxiv.org/pdf/2211.04894)


![Fig](figs/in_the_wild_on_kinetics.png)

Corresponding video results can be found [here](https://github.com/teowu/DOVER/figs).



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-konvid-1k)](https://paperswithcode.com/sota/video-quality-assessment-on-konvid-1k?p=disentangling-aesthetic-and-technical-effects)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-live-fb-lsvq)](https://paperswithcode.com/sota/video-quality-assessment-on-live-fb-lsvq?p=disentangling-aesthetic-and-technical-effects)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-live-vqc)](https://paperswithcode.com/sota/video-quality-assessment-on-live-vqc?p=disentangling-aesthetic-and-technical-effects)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=disentangling-aesthetic-and-technical-effects)


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
git clone https://github.com/teowu/DOVER.git 
cd DOVER 
pip install .  #stop here if you do not need pretrained weights, but why not?
mkdir pretrained_weights 
cd pretrained_weights 
wget https://github.com/teowu/DOVER/releases/download/v0.1.0/DOVER.pth 
cd ..
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
-- the technical quality of video [./demo/17734.mp4] is better than 43% of videos, with normalized score 0.10.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 64% of videos, with normalized score 0.51.
Compared with all videos in the KoNViD-1k dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 75% of videos, with normalized score 0.77.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 91% of videos, with normalized score 1.21.
Compared with all videos in the LSVQ_Test dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 69% of videos, with normalized score 0.59.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 79% of videos, with normalized score 0.85.
Compared with all videos in the LSVQ_1080P dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 53% of videos, with normalized score 0.25.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 54% of videos, with normalized score 0.25.
Compared with all videos in the YouTube_UGC dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 71% of videos, with normalized score 0.65.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 80% of videos, with normalized score 0.86.
```

## Evaluate on a Set of Unlabelled Videos


```shell
    python evaluate_a_set_of_videos.py -dir $YOUR_SPECIFID_DIR$
```

The results are stored as pickle files in dover_predictions with prefix `val-custom_`.


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
  journal={arXiv preprint arXiv:2211.04894},
  year={2022}
}
```