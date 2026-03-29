---
layout: ../../layouts/blogLayout.astro
title: "Data: The Practical Chain in Machine Learning"
description: "This introductory article would briefly explain the last piece of ML: Data. It is the most painstaking process in building models, but it's also the essential part of machine learning. We'll learn how data acts as a practical chain in machine learning that limits algorithms while giving meaning to the subject."
tags: ["data", "machine learning", "beginner", "workflow"]
date: "28 Mar, 2026"
---

# Data: The Practical Chain in Machine Learning

In my previous introductory posts, I covered two components of machine learning: optimizers (gradient descent) and models (neural networks). Now, it's time to learn the last piece of ML — Data.

If you ask machine learning experts about their favorite part of their work, the answers might vary. But if you ask them about the part they are most reluctant to do, the answer is definitely dealing with data. This is a weird phenomenon since we know machine learning is all about dealing with data. Then, how can someone who hate data become an expert? The answer, however, is complicated. So, in this post, let's dive into the most painstaking part of machine learning so that we can better understand the most fundamental part of it.

## Raw Data

What is data? This is a silly question, as we all have an idea about it. However, I'll still give you a precise definition to make it clearer. In machine learning, the term "data" refers to "**the raw set of observations, measurements, facts, or figures**". The key part here is "raw", which means that data is information that hasn't been processed yet. Machine learning is exactly how we process data, so data itself is the essential element of machine learning, or it's the reason why we need machine learning. 

Then, why do machine learning experts hate data?  

Again, it's because the data is "raw". But what does "raw" imply? 

In real life, nothing is perfect. Even in the 21st century, with so many advanced scientific discoveries and engineering results, "observations, measurements, facts, or figures" still cannot be flawless. The presence of error is inevitable. Just take a picture of anything around you using your phone, and compare it with what you see from your naked eye. Can you feel the difference? Of course. You might argue that it's because your phone's camera is cheap, but even if you use those expensive, professional, high-end cameras, you will still spot the difference. The history of science has told us that errors in measurements are ineliminable. When we apply this principle to data, it implies the fact that noise would always be present in data.

So what? We know there would be noise, but how would that matter? If the noise are small enough, to an extent that people can't even spot, then there is no perceivable difference. While this holds true in ideal cases, one must also consider the practical factors. High-precision measurements are expensive, while machine learning tasks are always data-hungry. It's just unrealistic to have all data being nearly perfectly measured in most situations.

Moreover, even your measurements (like taking photos) are nearly flawless, noise can come from not only the measurements, but also the content. Imagine that you are collecting data of dogs but you just carelessly put an image of a cat into your dataset. The cat image would be a noise no matter how perfect the image is. Similarly, watermarks, lighting and colors — these content-wise mistakes all eventually become noise in the dataset. Actually, in machine learning, everything that makes the dataset differ from reality is noise.

To answer the question, "raw" means that the data comes from reality directly with an unpredictable amount of noise.

## Cooking the data

Now we know how noisy can raw data be. But machine learning is designed for processing raw data, so why is dealing with data so painful? To answer this question, let's go through how we process data once so that we can understand the difficulties better while learning more about an ML workflow.

In practical ML workflow, data is the part that takes the most time. Machine learning experts need to spend countless hours on data. The process can be roughly separated into three parts, collection, cleaning and augmentation.

Data collection is the process of getting relevant data and creating a dataset. You can either do this by physical collection of data like doing experiments or by searching on the internet like web scraping. This stage would also include labelling in classification tasks, which means you might need to manually classify data.

Cleaning data is where you try to eliminate the noise we mentioned above. Usually done by checking through the dataset, either through looking at statistical metrics or manually inspecting data. For instance, you might really need to inspect individual images in an image dataset to avoid mislabeling. 

Augmentation happens before you use the data in training. You can expand the dataset in terms of quality and quantity by adding filters, rotations or cropping. This is usually done by an automated pipeline.

Among the three stages, augmentation is pretty easy, but collection and cleaning are tedious. If you have to create a massive dataset of dogs and cats, you would need to search for thousands of images online, labelling them and cleaning them. What kind of workload is that? More importantly, the process is brainless. You don't really need to think about anything. For most people, this is a mental torture. Imagine a machine learning expert sitting in front of a computer for months, but he or she is not devising architectures or training models, not even writing code, just cleaning data. Why experts hate dealing with data would then be understandable. When it's a large-scale industrial project, this work would be outsourced to over thousands of people, but when you are just writing a demo or having some small-scale projects, you'll need to suffer.

## Solutions?

The reason why experts hate dealing with data is explained, so let's get into some deeper discussion on data. We mentioned that machine learning is meant to deal with data, including noisy data. In fact, algorithms and architectures are designed for processing noisy data. One crucial target of a good architecture is to train on noisy data.

Let's think about this in a more fundamental way. What machine learning is trying to do is to learn a relationship or distribution shown in the dataset, but the actual goal is not like that. What machine learning tasks are trying to do is always about tackling a problem. For example, instead of learning the relationship between pixels and labels, the actual goal is classifying an image. Therefore, we can't just ignore everything and simply learn from the dataset.

Mathematically, it's really easy to just learn from a dataset, since there are countless interpolation methods that guarantee every data point can be predicted by the model. If we are satisfied with that, the subject "machine learning" is in fact meaningless. Why would we need neural networks if a random math equation can already give us the solution?

We can actually have a thought experiment. If we have infinite resources and abilities to collect data to any precision and any quantity, which means we can have all possible data of a problem. Then, do we need machine learning? For example, a model predicting the age of people by their faces. Since we already have all possible data, including the one being predicted, the "prediction" is more like an inquiry. We just need to find the information in the data like finding a book in libraries. We don't really need machine learning to learn a relationship between things; instead, we just need a powerful inquiry system.

Luckily, we don't have such ability in real life. The one being predicted is not in the dataset, and that's the point of learning. Just like the questions we encounter in the exam are never shown in the textbook. We need to be aware of the fact that machine learning is an engineering subject that focuses on not only the theory but also the practice.

Therefore, machine learning algorithms would focus on how to eliminate the effect of noise, and it thus introduces the most important ability a model should possess — generalization ability. It refers to the ability of a model to perform on unseen data that wasn't shown in the dataset. The generalization ability means model should try to ignore the irrelevant noise in the dataset and learn the determinate features from the data. It is one of the fundamental goals that machine learning algorithms are striving for.

However, that does not mean the quality of data doesn't matter. Just like you can't learn Chinese from an English book, data is still the bottleneck of models' performance. Even though scientists are trying their best to design architectures and methods that ensure the model's generalization ability, the inherent data problem is still catastrophic. That's why people have to spend most of their time on cleaning data. In most cases, obtaining a better dataset is the most effective way to improve model's performance. Again, machine learning is a practical subject, so you can't just assume everything being perfect to avoid the painstaking work. Data is just like chains on machine learning tasks, which force people to build models that are based on imperfect data. At the same time, imperfect data is the reason why we need machine learning. 

## How to test?

We learnt about the generalization ability of models, but how can we test that? It's another important part in doing machine learning research and developing practical apps, since an accurate estimation of the model's performance in reality is critical. In fact, I've faced scenarios that improper test methods lead to misleading results, and it's really something we should pay attention to.

The standard method is to conduct a data split. Splitting the dataset into training, validation and test sets, and train models only on the training set. The validation set is what you use to evaluate the models' performance during training and make adjustments on models. Although data in the validation set are not seen by the models during training, the models are still tuned and optimized with reference to it. Therefore, it's still not rigorous enough to evaluate the model's generalization ability on that. The correct way is to use the test dataset, which consists of data that are not involved in the training process at all. The set is only used when the final models is out as the ultimate test on the models' real performance. The actual application environment is also a test dataset, since no modification can be made.

It is of paramount importance to have appropriate and rigorous data splitting. Any mistakes could really cause fatal results. I'll have another post in the future to discuss this problem with my personal example.

## Conclusion

Data is the most painstaking part of machine learning. It requires countless hours being spent on repetitive, mindless, tedious work. Moving beyond the painful workload, data quality also limits the model's performance. With erroneous data, model can't really learn well, and it's just like chains on machine learning. Meanwhile, noisy data make machine learning meaningful and turn the subject into an engineering subject that focuses on solving practical problems. To tackle imperfect data, models have to possess generalization ability, and scientists have to design methods that improve and test generalization ability. Data would always be the inevitable yet fundamental component of machine learning.

In this post, we investigate the position of data in machine learning, workflow of data processing and a little bit about generalization ability. As an introductory post, it again omits many details. I'll have more detailed posts discussing these topics in the future, but for now, let's just stop here. 

Do leave a comment if you think this post helps you a little bit, or if you have anything you want to let me know.