# Neonatal pain detection

## Introduction

This project aims to detect neonates' pain by video based on the idea from this [paper](https://discovery.researcher.life/article/video-based-neonatal-pain-assessment-in-uncontrolled-conditions/9f7a31c43441321dbf7c97175d4ad900) :

```
Video-Based Neonatal Pain Assessment in Uncontrolled Conditions
```

The project is still developing. Up to now the facial part has been completed. 

## How to use?

You can create a dir named 'data' to put your different level videos in it. You should also create a dir named 'snapshots' to store the checkpoint info for model. Another dir named 'train_logs' is considered to store the tensorboard info during training.

Just run :

```shell
python3 main.py
```

 OR

```shell
python3 test_forward.py
```

to test model.

## How to get data?

The dataset of neonatal pain level video hasn't been found. If you have any dataset like this please email me(levi.xu.bin@gmail.com)