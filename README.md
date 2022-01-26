# A Second look at Exponential and Cosine Step Sizes: Simplicity, Adaptivity, and Performance
This repository contains PyTorch codes for the experiments on deep learning in the paper:

**[A Second look at Exponential and Cosine Step Sizes: Simplicity, Adaptivity, and Performance](https://arxiv.org/abs/2002.05273)**  
Xiaoyu Li*, Zhenxun Zhuang*, Francesco Orabona

### Description
Stochastic Gradient Descent (SGD) is a popular tool in training large-scale machine learning models. Its performance, however, is highly variable, depending crucially on the choice of the step sizes. Accordingly, a variety of strategies for tuning the step sizes have been proposed, ranging from coordinate-wise approaches (a.k.a. adaptive step sizes) to sophisticated heuristics to change the step size in each iteration. In this paper, we study two step size schedules whose power has been repeatedly confirmed in practice: the exponential and the cosine step sizes. For the first time, we provide theoretical support for them proving convergence rates for smooth non-convex functions, with and without the Polyak-Lojasiewicz (PL) condition. Moreover, we show the surprising property that these two strategies are adaptive to the noise level in the stochastic gradients of PL functions. That is, contrary to polynomial step sizes, they achieve almost optimal performance without needing to know the noise level nor tuning their hyperparameters based on it. Finally, we conduct a fair and comprehensive empirical evaluation of real-world datasets with deep learning architectures. Results show that, even if only requiring at most two hyperparameters to tune, these two strategies best or match the performance of various finely-tuned state-of-the-art strategies.

### Requirements
Run the following command to install required libraries:
```
pip install -r requirements.txt
```
