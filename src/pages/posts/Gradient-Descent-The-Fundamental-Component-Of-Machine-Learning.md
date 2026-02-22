---
layout: ../../layouts/blogLayout.astro
title: "Gradient Descent: The Fundamental Component Of Machine Learning"
description: "In this article, I will introduce one of the most important algorithms in machine learning that make everything possible"
tags: ["gradient descent", "machine learning", "beginner"]
date: "30 Jan, 2026"

---



# Gradient Descent: The Fundamental Component Of Machine Learning

In 2021, OpenAI released its new generative model ChatGPT 3.5. Since then, AI has become more popular than ever. Now, almost everyone, from schoolchildren to working adults, has heard of it. Meanwhile, you may have heard about machine learning, which is a subfield of AI that investigates how computers can learn from data and perform tasks without being programmed manually for every rule. Models like chatbots or image generators are typical examples. By letting machines learn from given data, we simplify AI development, as we no longer need to define every rule for machines. Owing to these advantages, many successful AI models are built on machine learning. But the simple term "machine learning" is confusing to many - How can machines learn like humans? In this article, I'm going to explain the theory behind that. It is not magic; it is ingenious algorithms created by computer scientists.

## What does AI actually mean?

Before diving into gradient descent, let's learn more about AI first. Over the past years, AI has been one of the most discussed concepts, although the actual definition of AI remains unclear to most people. Therefore, I believe some clarifications are essential before we talk about the details.

As we all know, AI stands for Artificial Intelligence, which means human-made things that possess "intelligence". The term AI does not refer to a single entity, but rather encompasses a range of practices from simple to complex. The origin of AI can be traced back to the 1950s, when Alan Turing, who was later called the father of AI, proposed the idea to simulate human intelligence by machines. From this definition, we can see the general purpose of AI: imitating human. Therefore, AI can be something simple, as long as it imitates humans. The NPC you fighting in PVE games, the bot you play with in tic tac toe, and the annoying customer service chatbot you talked with are all AI. AI does not mean it is "smart", it can be just a series of rules that attempt to simulate human behaviors. Given such a vague definition, naturally AI got a lot of branches that mainly differ from each other by the methods they use to simulate humans. Machine learning is one of them.

Unlike other types of AI that require manual programming, machine learning aims to let AI learn by itself. Moving beyond manual rule defining for computers, such as "Place the first piece in the middle", machine learning approaches just give computer a bunch of data, and ask it to learn by itself. By that, we avoid the complicated and often abstract (because you never know the best steps) programming, so all we need to do is focus on how to make machines learn better.

Additionally, when we talk about AI, there is always input and output. Therefore, identifying the input and output is fundamental  to any AI task. The input and output can be text and text like chatbot, they can be image and text like a classifier, or even a game move and game move in video games. It is possible to have multiple types of input, like videos are combinations of images and audio. Usually, AI models are explicitly trained for a certain combination of input and output.

Let's pause here. I know you probably have a lot of questions. The most intuitive one may be "How's that possible to make computer learn like human? Isn't it just made for arithmetic? " For the latter, you are absolutely right, computer are made for arithmetic - or more specifically, bit operation of binary numbers. And the former is exactly what I am going to explain: how computer scientists convert learning into equations for computers.

## Learning by numbers

To explain the process in a simple way, let's imagine we are computer scientists building an AI model, maybe a cat-vs-dog classifier. However, you would still need high-school-level math to understand the details of what we are doing.

Since computers can only process number, the first step is to convert everything into numbers, including both the input and output. This is quite easy nowadays as everything is already digital, images are expressed as matrices of numbers representing color, audios are sequences of number representing the frequency, and texts are just encoded numbers representing characters. In our classifier task, the input and output are clear, where the input is an image and the output is a class label. By converting it into numerical form we obtain a matrix as input and a number as output.

After digitalizing everything, we find that things become easier. Our task shifts from classifying images, which is vague and hard to find rules, to turning a matrix to a number, which sounds more like a solvable math problem we have encountered in school. Converting the unclear "learning" process to mathematical terms, we are essentially asking the computer to find a relationship between the numerical variables.

### The Simplest Case

The task might sound familiar to you. Finding relationship between numbers. Let's look at an easier example first, when we finished an experiment in science, we usually plot a graph:

<img src="\src\static\images\PixPin_2026-02-01_11-53-43.png" style="zoom: 50%; display: block; margin-left: auto; margin-right: auto;" />

draw a best-fit line:

<img src="\src\static\images\PixPin_2026-02-01_11-55-20.png" style="zoom: 50%; display: block; margin-left: auto; margin-right: auto;" />

and state the **relationship** between variables:

<img src="\src\static\images\PixPin_2026-02-01_11-55-39.png" style="zoom: 60%; display: block; margin-left: auto; margin-right: auto;" />

Yes, **relationship**, exactly the same thing we are trying to do now. In mathematics, the above process is called **regression**, which is the method to find relationship between independent variable and dependent variables. In fact, a simple regression like that can already be considered as machine learning, it's the fundamental form of **supervised machine learning**.

Maybe we can get some ideas from regression then, so let's check out how regression work. 

There are various ways to perform regression, but before that, let's make some clarification. The x (input) and y (output) data shown in the table, is what machine learnt from, namely the training data. The regression line, with equation y=ax+b is the model we obtained. Our goal is building a model that can predict the output from the given input as the relationship shown in the training data.

Now we can look at how regression actually work. A fundamental question for conducting regression is: "what is a good regression?". It might sounds like a stupid question, we all know a good regression is the one "fit data the best". But "best" is a vague word from human languages, and there is no a mathematical definition for "best". The one closest to average value can be the best, the one closest to endpoints, or the one pass through most data points. All these claims could be reasonable explanation of a good regression, but we must choose one to keep with. Different definition on the "best regression" creates different methods, and we'll focus on the most popular one.

One intuitive definition is the line that has a least overall distance with data points, which is the regression method used in many software, Ordinary Least Squares (OLS). In this regression method, we define a good regression as the one that has the minimum squared differences between predicted values and training data. A critical concept here is squared differences, which is defined as:
$$
\text{Squared Difference}=(y-\hat{y})^2
$$
where $y,\hat{y}$ are the predicted output and training output respectively. 

Unlike regular arithmetic differences from subtraction, squared differences guarantee the value must be positive so that values will not offset each others in summation. By summing up the squared differences, we obtained the Sum of Square Errors (SSE) of a regression, which is defined as:
$$
\text{SSE}= \sum_{i=0}^{n}{(y_i-\hat{y_i})^2}
$$
where n is the total amount of training data, $y_i,\hat{y_i}$ are the $i$ in in the training data and in predicted data respectively.

You would find that our goal becomes clear now. We are attempting to find a line that has a lowest SSE on our training data, which is an optimization problem. Great! Clear mathematical expression, and computer would love it.

Now we can use our math knowledge. In linear regression, we are attempting to achieve the minimum point of SSE, which is mathematically a function with the line as a variable. Meanwhile, the line are given by line equation $y=ax+b$, with two variables, gradient ($a$) and bias ($b$). Thus, SSE is actually a bivariable function and we are seeking for the best gradient and bias value. If you have done calculus in high school, I believe you will instantly think of using differentiation in optimization, where we find the points with zero gradient. But what we learnt in high school is for single variable, and here we got two variables. In math, the differentiation used for multivariable function is called partial derivative, which is obtained by differentiating in respect to one certain variable. I am not going to explain the math here, just treat it as differentiating one certain variable and take others as constant. By setting both derivative to zero, we can obtained the optimum value. The math steps are shown below
$$
\mathcal{L}(a,b) = SSE = \sum_{i=0}^{n}(y_i-\hat{y}_i)^2\\
\mathcal{L}(a,b) = \sum_{i=0}^{n}(y_i-(ax_i+b))^2\\
\frac{\partial \mathcal{L}}{\partial a} = \sum_{i=0}^{n}(2ax_i^2+2bx_i-2y_ix_i)\\
\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=0}^{n}(2ax_i+2b-2y_i)\\
\frac{\partial \mathcal{L}}{\partial a} =0\\
\frac{\partial \mathcal{L}}{\partial b} =0\\
\begin{cases}
\sum_{i=0}^{n}(2ax_i^2+2bx_i-2y_ix_i) = 0\\
\sum_{i=0}^{n}(2ax_i+2b-2y_i) = 0\\
\end{cases}
$$
From that, we end up with a system of simultaneous equations for the gradient and the bias, which can be easily solved with given data. Hereby, we finished drawing our "best" regression line - a simplest machine learning scenario.

### Getting More Complicated - Gradient Descent

But plot is for two variables only, in our case, we have thousands of variables (each pixel of image is a variable) with relationship that is much more complicated than linear, exponential or power. The task is no longer just drawing a straight line, we need to let computers find the complex hidden relationship that not even humans can find.

Recall how we work on regression task, we find a loss function, put our line equation into it and use multivariable calculus to find the optimum parameters. It's actually exactly how we cope with more complicated scenarios. 

I want to define the term "model" first. In linear regression, the model is a simple line equation $y= ax+b$, which gives us a line. Model refer to the mathematical framework we gave computer to train for, and it accounts for a large portion of the final performance. Imagine that we keep using line regression to make our classifier, asking computer to compute the best-fit line that map pixels to class label, ludicrous right? The choice of appropriate model is the foundation of all machine learning tasks, bad model choices would limit the final performance or even lead to failure. But again, one of the main aim of machine learning is reduce specific manual rule setting, so we want the machine to learn the pattern itself instead of giving them the answer. If we need to define a specific model to every task, like linear $y=ax+b$, quadratic $ax^2+bx+c$ or exponential $ae^{bx}+c$ , we are contradicting with the aim of machine learning. That's why we need some more universal model choice that fit for at least most situations, in deep learning, which is a subfield of machine learning, we use neural networks. 

We will not talk too much about neural networks here as the topic of this article is gradient descent, and I will post another article later to introduce and investigate more on neural networks. Here, just remember that we will need a more general way to express our models rather than a specific equation like $y=ax+b$. You might already notice that the key part of models in machine learning is the constants, which is called learnable parameters of a model. These parameters are fundamental components that makes a equation into a model. Therefore, we can express our model as a function with two variables, parameters and input, denoted as $M(x,w)$. Now we can use this generalized expression to investigate the essence of machine learning.

Again, look back at the process of linear regression above, we would find another key component: Loss Function. It gives a metric to measure the model's performance, and acts as a goal for learning. Similarly, loss function plays a vita role in machine learning and affects the result significantly. It's another field that is highly active and critical in research of computer science, a good loss function have to reflect the actual performance of models. I will have posts that talk about loss function and model evaluation more in depth later, now let's just clarify its general form first. Since loss function is measuring the performance of model on dataset, it takes two variable, model and dataset, to be more specific, it uses the output result of the model and compare with the   target results in the dataset. Thus, the expression of a loss function can be written as $\mathcal{L} (M(x,w),y)$, where $x,y,w$ are the input and output values in dataset and the wight respectively.

We can finally move on to the last piece of machine learning: optimizer, which is the calculus process in linear regression. You might ask why the calculus above becomes an "optimizer", isn't that just some math process? The thing is, linear regression is just too simple, some simple math can already lead to an optimum answer. Unfortunately, our nature is not that simple, relationships are barely linear or quadratic, it must be some equations we cannot solve analytically. For instance, we all can solve $x^2+2x-2=0$, but I bet you can't solve $\frac{xe^{\sin{x^{2/3}}} -\tan x^{9}}{\sqrt{2}x^5-x^3+x-\pi}=1.2$, some equation like that transcendent the ability of algebra. We cannot solve them by analytic approaches. But lucky that we got computers, solving them numerically is absolutely possible, but how? Well, there are many methods, but here we focus on the one helps with optimization problems: Gradient Descent.

Gradient Descent, is a mathematical algorithm that optimize parameters to minimize the function. It is the most popular method used in AI to train a model, its variants dominate all most all large-scale machine learning tasks. But the idea is actually simple. Imagine the loss function as a graph, just like what we plotted in math classes, then the graph is something like a mountain, high hills and lowlands. If we can easily know the map of the mountain, we can just locate the lowest point, just like in linear regression. However, what we face in most tasks are mountains that are super complicated, and we have no idea about the big picture. We just know how it looks like at the point we standing, but we need to find somewhere that is as lowest as possible. To deal with such an ordeal, we must utilize every information we have, so what do we have? The answer is gradient. We know how steep is the point we are standing, as we can calculate the differentiation at certain point numerically. The gradient shows the direction of the surface we are standing, for sure, to get lower we should go with the direction that is descending. By stepping toward the descending direction, we should be able to reach somewhere low. Congratulation!  You just figure out the core idea behind gradient descent.

We need to write the process mathematically so that computer can understand. let's start from the graph, which is the loss function  $\mathcal{L} (M(x,w),y)$ against weight $w$, in actual tasks this would be some high dimensional graph that humans can never imagine, but just keep with the mountain to make everything humane as they are essentially the same. Now we can move on to the gradient, as the graph is loss against weight, the gradient is thus, $\frac{\partial \mathcal{L}}{\partial w}$. You can further apply the chain rule to make it $\frac{\partial \mathcal{L}}{\partial M}\cdot \frac{\partial \mathcal{M}}{\partial w}$, but keep it simple is fine as we are just talking about the general picture. Since weight should actually be a series of numbers as a high dimensional vector, a more standard way to write the gradient would be $\nabla \mathcal{L}(w)$, the symbol $\nabla$ here is "nabla" which means a vector of derivatives with respect to all variables, i.e. all components of $w$. $w$ here should actually be $w_i$ which means a certain point we start with. With the gradient, we can find the next point we are standing on the loss graph, denoted as $w_{i+1}=w_i - \eta\nabla \mathcal{L}(w)$ , through minus a value that is opposite to the gradient, we can make sure the new loss $\mathcal{L} (M(x,w_{i+1}),y)$ is lower than the old one. $\eta$ here is the learning rate, it's a parameter of our gradient descent optimizer that tells how much we walk toward the descending direction. It's what we called a "hyperparameter", which is set by humans manually, unlike "learnable parameter" before. Tuning hyperparameters is another important part in training models, we will lean more about it in future articles. 

From the equation $w_{i+1}=w_i - \eta\nabla \mathcal{L}(w)$, we can see gradient descent is a process of iterations. We are keep iterating on the loss map to reach the minimum. To visualize this, the following diagram shows gradient descent on a quadratic function. 

<img src="\src\static\images\image-20260221153512100.png" style="zoom: 60%; display: block; margin-left: auto; margin-right: auto;" />

With the gradient getting smaller, the point moves slower and gradually get stable. This is when you know the training is finished.

In practice, gradient descent has many variants to cope with different situations, like Adam, SGD, Momentum. We will not go through these variants here, but they are all similar to the most basic gradient descent, just with extra design to improve the stability or efficiency.

### So how do machine learn?

Now we has explained gradient descent, so we can conclude how machine learning work. The term "learning" in ML is essentially "parameters updating".

The structure of the model provides a framework for learning, it defines all the parameters computers need to learn; the dataset acts as textbooks, telling computers what they should learn; and the loss function judges the result, suggesting the computer how to learn. What actually group all these component, and used by computers is optimizer. Computer use optimizer to find the parameter that lower the loss, and eventually get the optimal model. 

The process should work smooth and efficiently, humans should be able to model everything without writing out the rules. Indeed, the recent development has shown us the potential of machine learning, especially when there are sufficient data and computational resources. However, everything has a tradeoff, to ensure low human intervention in the learning, ML models become a total black box for human. We have no idea about how model predict things, no idea about what each parameter means, no idea about why it might fail. Computer scientists are striving to make ML models interpretable, but many model's phenomenon still can't be explained well. More importantly, from above we see ML models are essentially statistic models that make a regression to the dataset, there is no big difference between linear regression and large language model, it's just line equation is easier than transformers (we'll talk about it in the future). Gradient descent help machine "learn", but it is not the way human think "learning" is. But it doesn't mean gradient descent cannot be interpretate, there are new models made by scientist that use special architecture to ensure interpretability, we might go through those architecture later. 

For now, I hope you have a basic understanding on machine learning after reading this article. It's a beginner article so we omit many details, if you are interested to know more, you can check out other posts I have to learn more in depth knowledge.