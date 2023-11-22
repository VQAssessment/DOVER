# Data Release Note

_v1.0.0: merge_divide_and_maxwell_splits_

The DIVIDE-3k dataset was originally proposed in DOVER, as the first dataset with aesthetic and technical labels in addition to overall labels. In this dataset, we invite trained subjects to conduct a subjective study in-lab.

In the project [ExplainableVQA](https://github.com/VQAssessment/ExplainableVQA), we further invite the same group of trained subjects to provide 13 dimensions of **explanation-level** labels for the DIVIDE-3k (*i.e.* the MaxWell database). These subjects also further labeled around 1K new videos to expand the database.

As the videos in MaxWell is a superset of the videos in the DIVIDE-3k, after the internal discussion of our team, we decide to unify the `train` and `val` splits in the two datasets, and re-run DOVER and DOVER++ on the merged `DIVIDE-MaxWell` database (*a superset of DIVIDE-3k, each one labeled with aesthetic and technical, and overall perspective scores*) on the same official split as provided in MaxWell, *i.e.* 3634 training videos (80%), and 909 validation videos. The results of both variants are shown as follows, while the model checkpoint of DOVER++ (on the official train / test sets) will be uploaded soon to faciliate further research.

## Download Videos

Download the videos in [Hugging Face Datasets](https://huggingface.co/datasets/teowu/DIVIDE-MaxWell/resolve/main/videos.zip).

## Labels

The labels are provided here for the official [training set](../examplar_data_labels/DIVIDE_MaxWell/train_labels.txt) and official [validation set](../examplar_data_labels/DIVIDE_MaxWell/val_labels.txt)

## Training for DOVER++

To run DOVER++ (enhance end-to-end training with *Aesthetic+Technical+overall scores*), the scripts are as follows:

```shell
python training_with_divide.py --train train-dividemaxwell --val val-dividemaxwell
```

## Results on the Updated Train-Test Splits

As we have changed the train-test split, the results for FAST-VQA (technical branch of DOVER), DOVER and DOVER++ are also changed. See the following table for their results.

### Zero-Shot

#### VQA Approaches

- DOVER (pre-trained on LSVQ)

SROCC: 0.7477 | PLCC:  0.7546 | KROCC: 0.5510 

- FAST-VQA (*==technical branch in DOVER*, pre-trained on LSVQ)

SROCC: 0.7204 | PLCC:  0.7282 | KROCC: 0.5286 

- Aesthetic Branch in DOVER (pre-trained on LSVQ)

SROCC: 0.7184 | PLCC:  0.7293 | KROCC: 0.5249 

#### IQA Approaches

- SAQI (CLIP-ResNet-50)

SROCC: 0.5518 | PLCC: 0.5549 | KROCC: 0.3814

- NIQE 

SROCC: 0.2847 | PLCC: 0.3014 | KROCC: 0.2150

- Q-Boost (based on [Q-Instruct](https://github.com/Q-Future/Q-Instruct))

SROCC: 0.6821 | PLCC: 0.6923 | KROCC: 0.4949


### Fine-tuned

#### Baseline Methods

- VSFA (Li *et al*, 2019, trained on the training set of DIVIDE-MaxWell, only overall score used)

SROCC: 0.6671 | PLCC: 0.6784 | KROCC: 0.4875

- BVQA (Zhang *et al*, 2022, trained on the training set of DIVIDE-MaxWell, only overall score used)

SROCC: 0.7418 | PLCC: 0.7394 | KROCC: 0.5341

- FAST-VQA (Wu *et al*, 2022, trained on the training set of DIVIDE-MaxWell, only overall score used)

SROCC: 0.7798 | PLCC: 0.7819 | KROCC: 0.5868

#### Methods Proposed with DIVIDE or MaxWell

- MaxVQA (trained on the training set of DIVIDE-MaxWell, 16 dimensions used)

SROCC: 0.8044 | PLCC: 0.8131 | KROCC: 0.6098

- DOVER++ (trained on the training set of DIVIDE-MaxWell, 3 dimensions used)

SROCC: 0.8071 | PLCC: 0.8126 | KROCC: 0.6136

*All methods are based on training and the results main contain randomness.*