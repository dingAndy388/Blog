---
layout: ../../layouts/blogLayout.astro
title: "The Importance of Rigorous Data Splitting : A Hard Lesson"
description: "In this blog, I explore the importance of data splittitng, especially a rigorous one in research and development. I include my personal experience in my iGEM project, to see how serious data leakage can lead to misleading results and affect other people in the community."
tags: ["data", "machine learning", "research"]
date: "2 May, 2026"
---

# The Importance of Rigorous Data Splitting : A Hard Lesson

Last time, we covered through some basic knowledge about data in machine learning. Again, data is the crucial component in not only developing machine learning applications but also in frontier research. Therefore, the quality of data is paramount. But the quality here does not only refers to any single metrics, but rather encompasses a wide range of practices. The noises and representative of the data are parts of data quality, but the way we treat dataset could also affect the quality.

We briefly mentioned data splitting in the previous post. For those who are not familiarized with this idea, data splitting refers to the process of separating dataset into training, validation, and testing. We do that to avoid something called "overfitting", which is the loss of generalization ability we talked about in previous post. This might sound like a piece of cake, but some nuance in practice would really lead to unexpected result. In fact, I've been tricked by misleading results made by failed data splitting.

## What's data splitting and why we need that

We first explain some background knowledge so newcomers can follow. As abovementioned, data splitting is just separating the dataset into three, but how exactly is it and why we need that?

In the standard procedure, we separate dataset into:

- Training: The data being fed to model for training
- Validation: The data used to evaluate the model during training
- Test: The final exam of model. Used only once after everything is settled.

The reason behind is intuitive. As we know that models as universal approximators can easily learn all the data through memorizing instead of learning, we must use some data they haven't seen during training to evaluate them. The phenomenon of model memorizing all the data is called **Overfitting**. It's like the biggest enemy in machine learning studies, a large portion of efforts machine learning experts spent in on reducing overfitting.

Training and validation are two set that will be accessed frequently during training. We might need to know how exactly models are trained in practice to understand this. Data are not being trained for once, that's way too extravagant, instead, data are being iterated through the model for over hundreds of times. After each iteration of dataset (called an epoch), the model will be evaluated on the validation set. So when training, there will be two curves, one validation and one training. Usually, the graph can already indicate the existence of overfitting. To visualize this, I wrote some simple code that shows overfitting under extreme cases.,

```python
# The dataset, model, training configs here are all intentionally set to exterme cases to showcase overfitting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a sythetic dataset
np.random.seed(42)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
# noise
y = 2 * X + np.random.normal(0, 0.5, (100, 1))

# Split
X_train, y_train = X[:20], y[:20]
X_val, y_val = X[20:], y[20:]

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    verbose=0
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (Memorizing)', color='#1f77b4', lw=2)
plt.plot(history.history['val_loss'], label='Validation Loss (Failing)', color='#d62728', lw=2)

plt.yscale('log')
plt.title('Overfitting', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Log Scale)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

# Highlight the divergence
plt.fill_between(range(300), history.history['loss'], history.history['val_loss'], 
                 color='red', alpha=0.1, label='Generalization Gap')

plt.show()
```



The graph we got looks like this:

![](D:\repo\Blog\public\images\image-20260502233351254.png)

That's a pretty clear demonstration of how overfitting looks like. If your training curve looks like this, you know your model is going wrong. This is the power of validation set. However, the choice of validation set is not that strict, as we just need to ensure the data are not used in the previous iteration. There is a method called "cross validation" that would mix validation and training set together, each time keeps a portion out to maximize the use of data.

The one that have to be really strict and rigorous, is the test dataset. It's just like the final exam of model that only used after your model is finalized. **EVERYTHING**. You should not tune a single parameter of your model base on the result on test dataset. That is the difference from validation set, because even your manual modification of hyperparameters counts as a "learning" on the dataset. It have to been purely unseen data that acts as a mock of real application to evaluate the generalization ability of models. This is where many catastrophic results come from.

PS. Data splitting is just a way to measure overfitting, not a solution. There are many techniques that tackle with overfitting, which we might talk about in future articles.

## What makes a failed data splitting?

We've seen why we need data splitting, but it still looks easy. Why would violations, which is called data leakage, be so common? The answer is simple: **there are so many insidious ways that cause data leakage** 

You might believe that if we split the dataset into three pieces before training, we are all done. But the reality is much more complicated, which is why data leakage exists. Let me give you an example, can you identify whether the following is a proper data splitting?

> You are performing a medical classification task based on clinical images. Your team has contacted a hospital, which provides you a dataset of 10,000 clinical images taken from 1,000 patients. You performed the data splitting immediately after you had received the data, separating it into training(70%), validation(15%) and test(15%) randomly. During the training, your test dataset was being kept in a isolated file in your computer, so the model has never seen it before. Finally, after you finished training and tunning of model, you evaluate your final model on the test dataset, reaching over 90% accuracy. Then you hand over the model to hospital.

Can you spot the mistake? What if I told you the hospital reports that the model is clinically unusable after few days? The hint lays between the number of images and number of patients.

As the dataset is constituted by 10,000 images from 1,000 patients, in average 10 images are taken from a same patient. When the images are taken from a same patient, it might probably be repeated or just with minor rotation. Now, reexamine how the data splitting process is done by random, which implies that it's likely that images in validation and test dataset are from shared patients. Although the model hasn't seen test dataset before, it has been trained on really similar data taken from same patients. Therefore, the performance on the test dataset couldn't reflect the actual ability on unseen data at all. That's why the hospital said it's clinical unusable.

Luckily, when it comes to productive environments like hospital, actual application can easily debunk the illusion on improper test dataset. Although there might still be economical or physical damage (think about those patient being misdiagnosed by the model), the truth will out. But when it comes to the academia, the situation might be different, and this is from my personal experience.

## A single line in paper

What I was working on is also a medical project. I'm working for my school's iGEM (An international biology competition) team's software subgroup, which is developing a image classifier on clinical images of eczema (see more about my project on [our wiki](**https://2025.igem.wiki/hkssc/software**)). I am the main, pretty much the only developer working on machine learning. Of course, I've done extensive research on this topic to see if there are any similar work, and I found a few.

Studies have used a wide variety of methods, from SVMs to CNNs or even Transformers to tackle with dermatology diagnosis models. This is definitely a challenging field since datasets are very noisy. When it comes to specific eczema subtype task like I was dealing with, the situation is even worse. I could only get at most two thousands images that are extremely imbalance, the major class got over 1000 images while the smallest class got merely a hundred. Most studies that focuses on purely images classification usually obtained a result around 75% to 80%, some exceptional ones get over 85% with novel techniques. But their datasets usually have over ten thousands of images, so the results are quite acceptable. However, there is one very worth mentioning work we found, EczemaNet[^1], which also focuses on eczema classification. They only have 500 images in their dataset, using a beginner level CNNs with a few layers, but eventually obtained a result near 95%.

You might say this is obviously something wrong. But for me, as beginner at that time (also now), I would rather believe there are something ignored, since that is a published paper. In fact, this paper is the source of confidence during conceptualization of the project, it makes us believe building a model reaching over 90% of accuracy is possible. 

However, when I try to build one myself, my model can barely reach 50%. I believed that was due to the noise in dataset, because others' work has proven the feasibility. Therefore, I go to recollect data, manually cleaning the data, and spending months on improving the dataset quality. Eventually I still failed, training on CNNs does not work at all. Luckily I didn't give up, I locked into improving the architecture, trying out different methodologies that fit into the dataset we got (You can check out the wiki to see how I deal with such a extreme dataset, I'll probably write a post on that as well). Finally, I got a barely acceptable mode with about 75% result.

Nevertheless, I'm still confused about my result. Why others can reach a wonderful result with even less data and simper models? So I went back to read their paper word by word, and I found something reveal the truth.

![image-20260503171748804](D:\repo\Blog\public\images\image-20260503171748804.png)

*A figure of data pipeline in EczemaNet paper[^1]*

Can you spot the problem? It happens before the test split. The image augmentation process take place before data splitting. Image augmentation is the process of expanding dataset through methods like rotation, flipping, cropping etc. If the data splitting take place after that, the test dataset are basically a modified version of training set. The result obtained by that is meaningless. This explained why their model has such a exceptional result.

To further validate my hypothesis, I've done an experiment on my model and dataset. I manually creates data leakage like what [^1] has done. Then my model's performance just soared to over 90% from 75% (again, you can check out the detail on the wiki). Noted that this cannot be acted as a quantitative measurement of the errors here, as the result on a leaked test dataset tells nothing, the real performance could be even worse.

## A hard lesson

What did I learned from this experience? There are quite a few.

- Rigorous data splitting is IMPORTANT
- Be careful about the procedure in paper
- Doubt when the results are abnormal

What even worse is that, this is not a sole case. When I intentionally read through papers to check out their data pipeline, it's quite common that people augment data before splitting. Although for experts, these issues are easy to spot, but for beginners like me, such misleading results could really confuse us. If we take it serious, it's a kind of academic integrity that faking results.

That doesn't mean all results are fake, many studies are still using standard and rigorous data splitting procedures that reflect actual results, which should be praised. 

So I hereby call for everyone, no matter you're doing researches, or working on industrial projects, remember the importance of rigorous data splitting.

[^1]: Junayed, Masum Shah & Sakib, Abu Noman Md & Anjum, Nipa & Islam, Md Baharul & Jeny, Afsana. (2020). EczemaNet: A Deep CNN-based Eczema Diseases Classification. 174-179. 10.1109/IPAS50080.2020.9334929.



