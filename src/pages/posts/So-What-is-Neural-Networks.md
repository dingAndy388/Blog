---
layout: ../../layouts/blogLayout.astro
title: "So What are Neural Networks : And Why Architecture Matters"
description: "Neural networks are the most common architecture we use today. But what are 'architectures'? In this article, I'll use neural network examples to explain the essence of model architecture"
tags: ["neural networks", "machine learning", "architectures", "introductory" ,"beginner"]
date: "26 Feb, 2026"
---

# So What are Neural Networks : And Why Architecture Matters

Last time, I wrote an introductory post on machine learning, mainly focusing on gradient descent. In that post, I mentioned the term "model" as one of the important components in machine learning, but due to the space limitations, we didn't talk too much about it. In fact, architecture can be considered as the most important part in machine learning since it's the foundation of all AI models, and computer scientists are refining and devising new architectures over and over. So this time, let's have a closer and deeper look at neural networks, which are the most popular model architectures people use nowadays. In this introductory post on model architecture, I'll not only cover the general theory of architectures but also provide insights into the rations behind architectures.

## What is architecture

In many posts and courses, people would just start introducing architectures by listing the formulas, diagrams or codes. Of course, that is the fastest way we learn architecture, especially when the readers already have preliminary knowledge. But for beginners, this might not be the most intuitive way. Therefore, I'll start by introducing the essence of architecture to you.

What is model architecture? In my previous post, I simply summarized it as "a mathematical framework we ask computers to train on". This definition is just a general purpose of architectures, and the actual meaning of architectures is way more complicated than that.

To show the complexity in a more acceptable way, let's do it with examples.

## Multilayer Perceptron (MLP)

Multilayer Perceptron (MLP) is one of the earliest neural network architectures, which simply looks like this:

<img src="\images\mlp.png" style="zoom: 50%; display: block; margin-left: auto; margin-right: auto;" />

You might have seen it before as it's really iconic, basically the symbol of neural networks.

This architecture is pretty simple, each neuron (the circle in the diagram) is a feature, or mathematically a scalar component in the vector. The lines that lie between layers and connect neurons are maps that transform the previous layer into the next layer, which are mathematically matrices. The mapping is exactly where the machine "learns" through gradient descent. However, we all know matrices only perform linear mapping, and simply using layers of linear map gives only linear result. Therefore, a non-linear activation function was applied to the outputs of mappings to give non-linearity.

Eventually, the architecture turns the input (1st layer) into the output (the last layer) through a series of mappings learnt from the dataset and thus finish the modelling.

This architecture looks extremely easy, it's just expressing the output as the result of a series of non-linear mappings acting on the input. But this simple architecture is super powerful. Scientists have proven that this simple network can approximate any continuous function to any precision (search "Universal Approximation Theory" online to learn more). This means we can theoretically approximate any relationships between input and output using MLP, exactly the fundamental goal of machine learning. So it seems like we have already found the universal answer, but we know it's not the end. There must be some problems that hinder us.

Obviously, computation, optimization and data matter. A mathematical proof that says we can approximate any relationship doesn't mean we can actually find the optimal solution for variables in real life. Practically, we don't have unlimited computational resources and we don't have perfect data. Although MLP is already a universal approximator mathematically, we must consider the real life constraints. That's why we need architecture advancements to cope with these practical factors. From this point of view, MLP is not flawless.

The first problem is data structure. MLP takes only vectors, but our world has far more than vectors. Images are matrices, audios are sequences and videos are sequences of matrices. We can't just convert all these structures into vectors without losing information. When we use MLP in images, it would just forsake the essential spatial information as matrices have been flattened into vectors. This waste of critical information is fatal, especially when there are already so many real-life constraints.

The waste of information doesn't just come from data structure. The architecture itself is already ignoring a lot. In MLP, we make almost no assumption apart from the fact that there is a relationship between input and output. While in real life, it's not the case. For example, when reading texts, we know there is relationship between words and the context. So we can actually make an assumption that the next word is affected by the previous word in building text generation models. These kinds of assumptions make the model learn much easier as we provide some prior information on how features interact, while MLP doesn't really provide any.

From above, you might already be able to feel the power of architecture. It's never an arbitrary choice, but a sophisticatedly devised model that stems from the particular requirement in practical application.

Now, we can move on to some other architectures to learn more about data structure and assumptions in architecture design.

## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) is another architecture that was designed for images. The idea was first introduced by Kunihiko Fukushima in 1980 and further refined by Yann LeCun et al in the late 1980s to 1990s into the first practical CNNs: LeNet.

We've been talking about assumptions when designing architecture as the foreknowledge on how features interact. For processing images, we can actually make a few assumptions.

Translational invariance is one assumption we can make. When recognizing an object from images, the absolute position of the object in the images doesn't really matter. Like in the following image with many ugly handwritten digit 2, they're all digit 2 regardless of their position in the image. Both digits in the middle and the corner are 2. That's what we call translational invariance in image recognition tasks.

<img src="\images\222.png" style="zoom: 50%; display: block; margin-left: auto; margin-right: auto;" />

Locality is another assumption we can made. When looking at images, we know that the neighboring pixels are related and distant pixels are less related. For example, when we look at the following image of birds soaring in the sky, the black pixels of birds are interrelated, while the sky pixels further away are not really related to the birds.

<img src="\images\birds.png" style="zoom: 50%; display: block; margin-left: auto; margin-right: auto;" />

With he above two assumptions we made, we can work on the architecture. Translational invariance means that the model should not rely on the absolute position of pixels, and locality means the model should focus on a small range of pixels every time. These ideas lead to CNNs. We can first take it as a constrained version of MLP.

### Convolution Kernels

As translational invariance states, the architecture we have should ideally omit the absolute position of pixels. Therefore, unlike what MLP does by learning weights for every single component of vectors, our architecture should use a universal map to all features. Nonetheless, locality implies that such a map should have a certain range, so that the model is taking neighboring pixels as a whole when processing. With this design philosophy, we come up with convolution kernels.

How convolution blocks work is demonstrated by the following gif. ([source](https://commons.wikimedia.org/wiki/File:2D_Convolution_Animation.gif) [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/))

<img src="\images\conv.gif" style="zoom: 50%; display: block; margin-left: auto; margin-right: auto;" />

Then, we can replace many of the layers in MLP with convolution kernels to process images. This gives us CNNs. ([source](https://commons.wikimedia.org/wiki/File:Typical_cnn.png) [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/))

<img src="\images\cnns.png" style="zoom: 50%; display: block; margin-left: auto; margin-right: auto;" />

Convolution kernels need much less computation compared to MLP while taking the spatial information of images into account. With theses assumptions, we omit the irrelevant features, and force the model to learn the appropriate invariant mapping. This is the advantage we get from the architecture advancement.

### Limitation

Are CNNs a perfect architecture for images? The answer is still no. In fact, the assumption help reduce the complexity of tasks, but also oversimplify things in some cases. It makes CNNs hard to get long-range relationships, and fails in tasks that require absolute position of objects. However, this is why we are still working on architectures today. The problem CNNs have would need newer architectures to solve and there is no perfect architecture — it's just a game of balance. 

## Conclusion

In this introductory post on architecture, we quickly go through MLP and CNNs. The goal of this post is not to teach you how to code neural networks, it's about the meaning of architecture.

From the example of MLP and CNNs, you should be able to understand that architectures mean data structure and assumption. They're far more than just a random mathematical framework we choose, but rather are the result of careful consideration of practical limitations and analysis on the properties of tasks.

In a machine learning task, there are usually three areas: computation, architecture and data. Architecture is the most adaptive one. In most cases, we can't really change computation resources or data we own, but we can change our architecture. That's why architecture design is essential. Up to this point, there are still a lot of problems we haven't solved yet. The models are mostly black boxes that we don't really know what's going on, training on noisy data is still hard and the performance of lightweight models is still limited. These problems all need better architectures to solve.

Unfortunately, we didn't cover the math and codes today due to space limitations. We didn't talk about the models dealing with sequences, which is the foundation of the language models we see today. However, I guess it's good to have a general understanding on architectures first, so we can dive into specific architectures in the future. In my next posts, I will continue introductory articles for beginners, and start to have some more in-depth blogs.

Do leave a comment if you think this post helps you a little bit, or if you have anything you want to let me know.