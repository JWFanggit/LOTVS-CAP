# LOTVS-Cognitive Accident Prediction
We are warmly to release a new benchmark on Accident Prediction in dashcam videos. The benchmark is called as CAP-DATA, which consists of 11,727 videos with 2.19 million frames. The fact- effect-reason-introspection description for each accident video is annotated, which consists of the factual description before accident, categorical accident description, accident reason description and preventive advice description. To our best knowledge, it is the largest accident prediction benchmark in driving scenarios.

In addition, we also propose a new model to fulfill a Cognitive Accident Prediction (CAP), which explores the human cognition clues to assist accident prediction in driving videos. Particularly, we explore the text description before accident and the driver attention in driving situations to boost the explainability of accident prediction model, which has the apparent semantic guidance for the object to be involved in accident and helps to find the crashing object efficiently. Extensive experiments shows that the proposed method can obtain larger Time-to-Accident (TTA) than other state-of-the-arts.


![image](https://github.com/JWFanggit/LOTVS-CAP/blob/main/CAP-DATA.png)

# Download Benchmark
we will publish our benchmark (CAP-DATA) soon.

# Implementation Details
# Training
Implementation details:
The code is implemented by Python 3.9 with the Pytorch platform. 

We use the bert model to encode the text.Download the [bert-base-uncased-pytorch_model](https://pan.baidu.com/s/1vnPIOLn7s_4MZyISjP5a0A)(Extraction code：rd4y)

MINI-Train: Download the training sample configuration [Here](https://pan.baidu.com/s/1SOLOM01OMlZSz5a7s2khHA )(Extraction code:ka5z)
   
a.Download the [DADA-1000](https://pan.baidu.com/s/1bLQb3Lz5atz6sgBz-VWNJQ) Extraction code：：472b).

b.Make data structure
>[rootpath]
>>[training]

>>>[rgb_videos]

>>>[focus_videos]

>>>[training.txt]

c.```Run Train.py```

FULL-Train:Download the training sample configuration [Here](https://pan.baidu.com/s/1Ls_qZZU_IMl6D8Muu7cMVg )(Extraction code:0zya)
                  
a.Download the DADA-2000 (can be downloaded [here](https://pan.baidu.com/s/1oxoQKYIaNCkLCxVCrOwgHw?pwd=ahyz)(Extraction code:ahyz))

b.Make data sturcture same as MINI-Train

c.```Run Train.py```

# Testing
Implementation details:

MINI-Test(DADA-2000-TEST):Download the inference model on the MINI-Test evaluation [Here](https://pan.baidu.com/s/1vcdTEn1g0EdWtastLsU8QA)(Extraction code:qe4w)

a.Use the DADA-1000 dataset.Make the inference model path place the ckpt_path at ```Test.py```

b.Make data structure
>[rootpath]
>>[testing]

>>>[rgb_videos]

>>>[testing.txt]

>>>[testing-text.txt]


c.```Run Test.py```

FULL-Test(CAP-DATA-TEST):Download the inference model on the FULL-Test evaluation [Here](https://pan.baidu.com/s/13iFDdi_aInqQBFOJHOXl8w)(Extraction code:keh4)

a.Use the CAP-DATA dataset.Make the inference model path place the ckpt_path at ```Test.py```

b.Make data structure same as MINI-Test.You can choose any of the txt documents in training sample configuration(FULL-Train-test) to replace the testing.txt to test

c.```Run Test.py```

If you use this dataset and the code, please cite the following bibtex format.
```
@article{DBLP:journals/corr/abs-2212-09381,
  author    = {Jianwu Fang and
               Lei{-}Lei Li and
               Kuan Yang and
               Zhedong Zheng and
               Jianru Xue and
               Tat{-}Seng Chua},
  title     = {Cognitive Accident Prediction in Driving Scenes: {A} Multimodality
               Benchmark},
  journal   = {CoRR},
  volume    = {abs/2212.09381},
  year      = {2022},
}
```
# Reference

Note: CAP-DATA benchmark can only be utilized for research. If you are interested in this work and the benchmark, please cite the work with following bibtex.

[2] Wentao Bao, Qi Yu, Yu Kong: DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation. ICCV 2021: 7599-7608


