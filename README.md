# LOTVS-CAP
The proposed CAP model consists of three core modules:attentive text-to-vision shift fusion module, attentive semantic context transfer module, and driver attention guided accident prediction module. It is worth noting that each module explores the attention mechanism and contributes a brain inspired cognitive modeling. We learn the coherence of text-vision semantics and the activated scene context by cascade attentive networks, where each attentive module fulfills the core semantics learning for accident prediction. Attentive text to-vision shift fusion module is modeled by inferring the coherently semantic relation of text and video for accident prediction. Besides, text-to-video shift strategy aims to leverage the semantic knowledge in text to video, so as to adapt to the testing phase only with the video data in practical use. Attentive semantic context transfer module encodes the scene context that is modeled by the Graph Neural Network (GNN) and Gated Recurrent Unit (GRU),
which imitates the ability of human-beings for historical and contextual memory learning. Then, the driver attention guided accident prediction module predicts the beginning time of road collision and reconstructs the driver attention map simultaneously. Exhaustive experiments shows that the proposed method can obtain larger Time-to-Accident (TTA) than other state-of-the-arts.
![image](https://raw.githubusercontent.com/JWFanggit/LOTVS-CAP/main/model.png)
# Testing
Min-Test(DADA-2000-TEST):Download the best model [Here](https://pan.baidu.com/s/1tgXcaEaWQdgmoB7eubuZfA)(Extraction code：i8mg)

Full-Test(CAP-DATA-TEST):Download the best model [Here](https://pan.baidu.com/s/13iFDdi_aInqQBFOJHOXl8w)(Extraction code:keh4)
# Dataset download:
We are worm-hearted to release this benchmark here, and sincerely invite to use and share it. Our CAP-DATA can be downloaded from （）This is the dataset that we re-uploaded after sorting out. At the same time, the complete DADA-2000 dataset can be download from（）.The DADA-Small dataset you can follow  the [work](https://github.com/Cogito2012/DRIVE.git) .

Note: CAP-DATA benchmark can only be utilized for research. If you are interested in this work and the benchmark, please cite the work with following bibtex.
