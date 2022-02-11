# Detecting Difficulty of Math Problems
![](mathematics.jpeg)

## Project Description

**Members: Tonyaradzwa Chivandire, Salih Erdal**

This project will focus on training a neural network (NN) to detect the difficulty level of a math problem and, in turn, generate its own difficult math problem. 

## Introduction

Can a computer learn to identify the difficulty level of a math problem? This question motivates our study, and we seek to explore the computer's ability to determine the difficulty of problems in different areas of math and its ability to generate its own difficult problem by utilizing neural network architectures.

The difficulty in teaching computers to detect difficulty is identifying what makes the problems difficult—problems could require immense calculation or many complex ad-hoc strategies applied in a certain order. Such parameters, are potential inputs to our neurons. 

Our approach will combine the strengths of previous approaches that train a computer to solve math problems.

We have to deal with limited computing resources, which might potentially decrease the accuracy of our model by affecting the size of our training data set.

We expect our model to perform with 70% accuracy averaged over all areas of mathon interpolation estimates and 60% on extrapolation estimates.  

## Dev Stages

**stage 1:**
  - input: problem
  - output: difficulty of problem

* where is data coming from
  * https://github.com/deepmind/mathematics_dataset
* how will we label training data?
  * they are already labeled

interpolation: our test data

extrapolation: other data

**stage 2:**

input: difficulty level

output: random question at that difficulty level with its solution

## Ethical Sweep

The model that we will be training may be used for good purposes, such as generating problems appropriate to the level of a student who needs more practice on some types of questions, and evaluating the student's answer. If our model or future work can find solutions to problems that are not easily solvable by humans, it might lead to further discussions about the nature of some problems.  



## Project Goals
1. Determine our parameters of difficulty.
2. Train the NN to generate a problem-solution pair that matches our standard of difficulty.
