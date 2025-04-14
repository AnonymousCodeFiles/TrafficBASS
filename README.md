

# TrafficBASS: A Boundary-Aware Active Learning Framework for Endpoint Advanced Persistent Threat Detection



**The repository of TrafficBASS, a lightweight endpoint APT detection framework.**

<div align="center">
<img src="https://github.com/AnonymousCodeFiles/TrafficBASS/images/Background.png" alt="Background" style="zoom:25%;" />
</div>  




<p align="center">Fig. 1 Background</p>


# Contents

- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Dataset and preparation](#Dataset-and-preparation)
- [Reproduce TrafficBASS](#Reproduce-TrafficBASS)
- [Acknowledgement](#Acknowledgement) 

# Introduction  

As illustrated in Figure 1,TrafficBASS tackles the challenges of APT long-tail distribution, concept drift and the need for lightweight, continuously learning models by leveraging the following key components: a Learnable Feature Projection (LFP) module, a Boundary-Aware Sampling Strategy (BASS) and an Adaptively Lightweight Fine-tuning (ALF) strategy directs ALBERT-based pre-training and adjusts LoRA rank size during fine-tuning through an innovative LFP attention adjustment mechanism to efficiently adapt to new threats over time while maintaining low computational overhead. The framework’s architecture is depicted in Fig. 2.

__Modules of TrafficBASS include:__

* __Learnable Feature Projection (LFP) __.
  This module dynamically integrates statistical and temporal features through adaptive feature adjustment mechanisms.
* __Boundary-Aware Sampling Strategy (BASS)__.
  This module effectively identifies the most informative samples through dynamic scoring, elastic memory banking, and adversarial expansion augmentation.
* __Adaptively Lightweight Fine-tuning (ALF) strategy.__
  ALF using ALBERT-based models with LFP attention LoRA to efficiently adapt to emerging threats while maintaining minimal computational requirements.



<div align="center">
<img src="https://github.com/AnonymousCodeFiles/TrafficBASS/images/TrafficBASS.png" alt="TrafficBASS" style="zoom:25%;" />
</div>  




<p align="center">Fig. 2 Overview of TrafficBASS</p>



# Requirements

Before using this project, you should configure the following environment.  

1. Requirements

```
python >= 3.8
transformer = 4.44.2
pytorch = 2.3.0+cu118 
torchvision == 0.18.0
torchaudio == 2.3.0
tokenizers == 0.19.1
peft = 0.11.1
```

2. Basic Dependencies

```
numpy == 1.24.3
pandas = 2.0.3
scikit-learn
huggingface-hub 
tqdm
```

3. Others

```shell
Ubuntu 20.04
```



# Dataset and preparation

1.Dataset 

We use the open source  [USTC-TFC](https://github.com/yungshenglu/USTC-TK2016 "USTC-TFC")  endpoint dataset and [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset "UNSW-NB15") IoT attacks dataset.

2.Pcap to side-channel features

You need to use split TCP pcaps into flow sessions. Then, using flowcontainer tool to extract time series features with the maximum packet length at 1600 and set the minimum session length to 20.

```shell
# run pcap2TCP.py to filliter TCP flows
# modify: theh path of 'path_pcap', 'path_wireshark' and 'path_save_pcap' to generate TCP flows.
python pcap2TCP.py
# Extract side-channel features from flow sessions
python session2features.py
# Generate train, test and pre-training data
python train_test_data.py
-> tensor(train, test), txt(pre-training)
```



# Reproduce TrafficBASS

## Pre-training

If you want to train a new TrafficBASS in your own scenario, you should firstly run the `train_test_data.py` to generate pre-training data (`file.npz`) by your own network traffic. And then put the generated `file.npz` into the training folder, and create a new folder to save `vocab`, `config`,`model.safetensors`  and `training_args.bin`. Finally, set the `Config` and run the `pre-training.py` for TrafficBASS pre-training.

```python
# Create new folder to save pre-traing model


'''
# Run
python pre-training.py

```

In addition, if you want to change the params to suit for your network environment, you could modify the params as follows.

```python
# Modified params
class ActiveLearningConfig:
    # base
    initial_labeled_samples: int = 10
    query_size: int = 100
    max_iterations: int = 100
    target_accuracy: float = 99.99
    random_seed: int = 42
    test_size: float = 0.2
    batch_size: int = 500
    
    # Params of BASS
    buffer_size: int = 5000
    memory_size: int = 1000
    tau: float = 0.1
    eps: float = 0.01
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    temperature: float = 1.0
    
# ...
```





# Acknowledgement

Thanks for these awesome resources that were used during the development of the TrafficBASS：  

* https://huggingface.co/
* https://github.com/yungshenglu/USTC-TK2016
* https://research.unsw.edu.au/projects/unsw-nb15-dataset
