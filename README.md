# LOTVS-Cognitive Accident Prediction
We are wormaly to release a new benchmark on Accident Prediction in dashcam videos. The benchmark is called as CAP-DATA, which consists of 11,727 videos with 2.19 million frames. The fact- effect-reason-introspection description for each accident video is annotated, which consists of the factual description before accident, categorical accident description, accident reason description and preventive advice description. To our best knowledge, it is the largest accident prediction benchmark in driving scenarios.

In addition, we also propose a new model to fulfill a Cognitive Accident Prediction (CAP), which explores the human cognition clues to assist accident prediction in driving videos. Particularly, we explore the text description before accident and the driver attention in driving situations to boost the explainability of accident prediction model, which has the apparent semantic guidance for the object to be involved in accident and helps to find the crashing object efficiently. Extensive experiments shows that the proposed method can obtain larger Time-to-Accident (TTA) than other state-of-the-arts.


![image](https://github.com/JWFanggit/LOTVS-CAP/blob/main/model.png)


CAP-DATA contains two parts: one is the DADA-2000 (can be downloaded [here](https://pan.baidu.com/s/1oxoQKYIaNCkLCxVCrOwgHw?pwd=ahyz)(Extraction code:ahyz)) which contains the driver attention for each frame and the text description, and another part with 9727 dashcam videos and text description (can be downloaded [here](https://pan.baidu.com/s/1QjrTiBEVLgwBPnGkBOKaZg?pwd=pcz5 )(Extraction code:pcz5)). 

In this work, we provide two cases of training and testing evalution: MINI-Train-Test and FULL-Train-Test. The training and testing set of MINI-Train-Test is the same as the work [2].

# Training
MINI-Train: Download the training sample configuration [Here](https://pan.baidu.com/s/1tgXcaEaWQdgmoB7eubuZfA)(Extraction code:i8mg)

FULL-Train(CAP-DATA-Train):Download the training sample configuration [Here](https://pan.baidu.com/s/13iFDdi_aInqQBFOJHOXl8w)(Extraction code:keh4)



# Testing
MINI-Test(DADA-2000-TEST):Download the inference model on the MINI-Test evaluation [Here](https://pan.baidu.com/s/1tgXcaEaWQdgmoB7eubuZfA)(Extraction code:i8mg)

FULL-Test(CAP-DATA-TEST):Download the inference model on the FULL-Test evaluation [Here](https://pan.baidu.com/s/13iFDdi_aInqQBFOJHOXl8w)(Extraction code:keh4)

# Reference

Note: CAP-DATA benchmark can only be utilized for research. If you are interested in this work and the benchmark, please cite the work with following bibtex.

[2] Wentao Bao, Qi Yu, Yu Kong: DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation. ICCV 2021: 7599-7608


