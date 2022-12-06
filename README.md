# LOTVS-Cognitive Accident Prediction
We are warmly to release a new benchmark on Accident Prediction in dashcam videos. The benchmark is called as CAP-DATA, which consists of 11,727 videos with 2.19 million frames. The fact- effect-reason-introspection description for each accident video is annotated, which consists of the factual description before accident, categorical accident description, accident reason description and preventive advice description. To our best knowledge, it is the largest accident prediction benchmark in driving scenarios.

In addition, we also propose a new model to fulfill a Cognitive Accident Prediction (CAP), which explores the human cognition clues to assist accident prediction in driving videos. Particularly, we explore the text description before accident and the driver attention in driving situations to boost the explainability of accident prediction model, which has the apparent semantic guidance for the object to be involved in accident and helps to find the crashing object efficiently. Extensive experiments shows that the proposed method can obtain larger Time-to-Accident (TTA) than other state-of-the-arts.


![image](https://github.com/JWFanggit/LOTVS-CAP/blob/main/CAP-DATA.png)

# Download Benchmark
CAP-DATA contains two parts: one is the DADA-2000 (can be downloaded [here](https://pan.baidu.com/s/1oxoQKYIaNCkLCxVCrOwgHw?pwd=ahyz)(Extraction code:ahyz)) which contains the driver attention for each frame and the text description, and another part with 9727 dashcam videos and text description (can be downloaded [here](https://pan.baidu.com/s/1QjrTiBEVLgwBPnGkBOKaZg?pwd=pcz5 )(Extraction code:pcz5)). 

In this work, we provide two cases of training and testing evalution: MINI-Train-Test and FULL-Train-Test. The training and testing set of MINI-Train-Test is the same as the work [2].

# Implementation Details
# Training
Implementation details:
The code is implemented by Python 3.9 with the Pytorch platform. 

MINI-Train: Download the training sample configuration [Here](https://pan.baidu.com/s/1SOLOM01OMlZSz5a7s2khHA )(Extraction code:ka5z)

a.Download the [DADA-1000](https://pan.baidu.com/share/init?surl=RfNjeW0Rjj6R4N7beSTYrA)(Extraction code:9pab)

b.Make data structure
>[rootpath]
>>[training]

>>>[rgb_videos]

>>>[focus_videos]

>>>[training.txt]

c.```Run Train.py```

FULL-Train(CAP-DATA-Train):Download the training sample configuration [Here](https://pan.baidu.com/s/1Ls_qZZU_IMl6D8Muu7cMVg )(Extraction code:0zya)

a.Download the DADA-2000 (can be downloaded [here](https://pan.baidu.com/s/1oxoQKYIaNCkLCxVCrOwgHw?pwd=ahyz)(Extraction code:ahyz))

b.Make data sturcture same as MINI-Train

c.```Run Train.py```

# Testing
Implementation details:


MINI-Test(DADA-2000-TEST):Download the inference model on the MINI-Test evaluation [Here](https://pan.baidu.com/s/1tgXcaEaWQdgmoB7eubuZfA)(Extraction code:i8mg)

a.Use the DADA-1000 dataset.Make the inference model path place the ckpt_path at ```Test.py```

b.Make data structure
>[rootpath]
>>[testing]

>>>[rgb_videos]

>>>[testing.txt]

c.```Run Test.py```

FULL-Test(CAP-DATA-TEST):Download the inference model on the FULL-Test evaluation [Here](https://pan.baidu.com/s/13iFDdi_aInqQBFOJHOXl8w)(Extraction code:keh4)
a.Use the CAP-DATA dataset.Make the inference model path place the ckpt_path at ```Test.py```

b.Make data structure same as MINI-Test.You can choose any of the txt documents in training sample configuration(FULL-Train-test) to replace the testing.txt to test

c.```Run Test.py```


# Reference

Note: CAP-DATA benchmark can only be utilized for research. If you are interested in this work and the benchmark, please cite the work with following bibtex.

[2] Wentao Bao, Qi Yu, Yu Kong: DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation. ICCV 2021: 7599-7608


