---
title: "Classifying images with NN from scratch"
subtitle: "Using numpy only and plenty of visualizations"
author: "Maciej Dobrzyński"
description: "A tutorial to classify images using a simple neural network written from scratch using only numpy"
institute: "IZB UniBe"
date: "24/08/2024"
abstract: "YAML"
keywords: 
  - machine learning
  - image classification
  - tutorial
---

# Intro

This tutorial demonstrates how to train a neural network classifier to differentiate between images of horizontal and vertical lines. 
The neural network is coded from scratch using only Python's numpy library.
We go through all the steps of the classic machine learning training loop comprising the forward step, backpropagation, and parameters update. 
The model will be evaluated on two validation sets.

The tutorial is inspired by S. Zhangs's [tutorial on Kaggle](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras) [1] and is based on [resources](#further-reading) [2]-[5].

# The data

The input data $x^{[0]}$ comprises 24 3x3 pixel images grouped into two classes: vertical and horizontal lines. Each pixel assumes a grayscale value between 0 and 1:

![png image](figs/training-set.png "Training set")

# The network

Our neural-network has a two-layer, fully connected architecture. 


![svg image](figs/nn.svg "NN scheme")

The scheme drawn with [NN-SVG](https://alexlenail.me/NN-SVG/index.html).

# Forward propagation

To calculate the output distribution $A^{[2]}$ from the input data $x^{[0]}$, we use the following equations:

$$
\begin{align}
z^{[1]} &= W^{[1]}  x^{[0]} + b^{[1]}\\
A^{[1]} &= \text{ReLU}(z^{[1]})\\[6pt]
z^{[2]} &= W^{[2]}  A^{[1]} + b^{[2]}\\
A^{[2]} &= S(z^{[2]})
\end{align}
$$


# Training

The animation below shows parameter changes during the training.
Warm and cold colors corresponds to positive and negative values, respectively. 
Gray scale colors are used for matrices with only positive values; black corresponds to 0 and white to 1. 

The prediction matrix $A^{[2]}$ visualizes class probabilities calculated for every example in the input dataset. 
Red and green dots indicate incorrect and correct predictions, respectively.

![gif image](figs/train-anim.gif "Training")


# Further reading

[1] Samson Zhang, _Simple MNIST NN from scratch (numpy, no TF/Keras)_, Kaggle, 2018; [HTML link](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras).

[2] Jeremy Howard and Sylvain Gugger, _Practical Deep Learning for Coders: Chapter 17 - A Neural Net from the Foundations_, fast.ai; [HTML link](https://fastai.github.io/fastbook2e/foundations.html).

[3] Shivam Mehta, _Deriving categorical cross entropy and softmax_, 2023; [HTML link](https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/).

[4] Eli Bendersky, _The Softmax function and its derivative_, 2016; [HTML link](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/).

[5] Katanforoosh & Kunin, _Initializing neural networks_, deeplearning.ai, 2019; [HTML link](https://www.deeplearning.ai/ai-notes/initialization/index.html)
