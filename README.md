# Detecting Difficulty of Math Problems
**Members: Tonyaradzwa Chivandire, Salih Erdal**
![](mathematics.jpeg)

## Project Description

This project will focus on training a neural network (NN) to detect the difficulty level of a math problem and, in turn, generate its own difficult math problem. 

## Introduction

Can a computer learn to identify the difficulty level of a math problem? This question motivates our study, and we seek to explore the computer's ability to determine the difficulty of problems in different areas of math and its ability to generate its own difficult problem by utilizing neural network architectures.

The difficulty in teaching computers to detect difficulty is identifying what makes the problems difficult—problems could require immense calculation or many complex ad-hoc strategies applied in a certain order. Such parameters, are potential inputs to our neurons. 

Our approach will combine the strengths of previous approaches that train a computer to solve math problems.

We have to deal with limited computing resources, which might potentially decrease the accuracy of our model by affecting the size of our training data set.

We expect our model to perform with 70% accuracy averaged over all areas of mathon interpolation estimates and 60% on extrapolation estimates. 

## Related Works
[**Text2Math: End-to-end Parsing Text into Math Expressions**](https://arxiv.org/abs/1910.06571)
(Yanyan Zou, Wei Lu. 2019)

Semantic parsing and solving of arithmetic word problems and equations, using end-to-end parsing. The model achieved up to 86.5 % accuracy for arithmetic word problems and up to 74.5% accuracy for equation parsing.

[**Using neural networks to solve advanced mathematics equations**](https://ai.facebook.com/blog/using-neural-networks-to-solve-advanced-mathematics-equations/)
(Facebook AI Blog)

Training a neural network to solve advanced math equations using seq2seq transformer model and a dataset of millions of math problems (problem-solution pairs). The model was very accurate, achieving 99.7% percent accuracy in solving integration problems.

[**ANALYSING MATHEMATICAL REASONING ABILITIES OF NEURAL MODELS**](https://openreview.net/pdf?id=H1gR5iR5FX) (David Saxton et al.. 2019)

This paper examines the capability of some Neural Network models with sequence-to-sequence architectures in developing mathematical reasoning abilities in order to solve high school difficulty math problems. The paper discusses how data(problem-solution pairs) was generated synthetically and notes the models were trained with free form textual input/output form, and compares the performance of these models.

[**Solving Math Equations with Neural Networks**](https://ai.plainenglish.io/solving-math-equations-with-neural-networks-f015351995e8)(Blog post)

This blog post shows a simple implementation of a recurrent neural network which learns the symbolic representation of numbers and algebraic operations. The model takes as input simple arithmetic math operations as strings. This source could be useful as a starting point for implementing a similar architecture with our own data.

## Dev Stages

**Stage 1:**
  - input: problem
  - output: difficulty of problem

* where is data coming from
  * https://github.com/deepmind/mathematics_dataset
* how will we label training data?
  * they are already labeled

interpolation: our test data

extrapolation: other data

**Stage 2:**

input: difficulty level

output: random question at that difficulty level with its solution

## Ethical Sweep

The model that we will be training may be used for good purposes, such as generating problems appropriate to the level of a student who needs more practice on some types of questions, and evaluating the student's answer. If our model or future work can find solutions to problems that are not easily solvable by humans, it might lead to further discussions about the nature of some problems.  
