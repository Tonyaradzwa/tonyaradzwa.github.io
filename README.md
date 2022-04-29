# Solving Difficult Math Problems
**Members: Tonyaradzwa Chivandire, Salih Erdal**
![](mathematics.jpeg)

## Project Description

This project will focus on training a neural network (NN) to solve math problems that are given in text form. 

## Introduction

Can a computer learn to solve math problems? This question motivates our study, and we seek to explore computers' problem-solving abilities in different areas of math by utilizing neural network architectures, e.g Recurrent NNs. 

The difficulty in teaching computers how to solve problems lies mainly in the fact that computers cannot "digest" a problem in the same way that humans do. When humans encounter problems, they extract the relevant information from what they are given; they decipher the pragmatic meaning of the terms to identify what the question is asking and apply a sequence of steps to arrive at a solution by using techniques that are not immediate in the questions. This is called mathematical reasoning, and whether computers can learn mathematical reasoning forms the basis of our inquiry. 

Before going into what exactly will constitute our research, we present a brief overview of the current literature. The area of using ML algorithms to teach computers how to solve math quesitons is relatively new, since it seems to be quite a challenging task that isn't very clearly defined. Since solving all kinds of math problems is not conceivable, current research usually focuses on solving problems in specific areas of math. Consequently, there are different methods tried for solving problems in different fields. One of the seemingly easier kinds of problems that people have focused on as proof of concept are arithmetic word problems. For example,[**Text2Math: End-to-end Parsing Text into Math Expressions**](https://arxiv.org/abs/1910.06571) by Yanyan Zou, Wei Lu. 2019 focuses on parsing arithmetic word problems and equations in a tree structure, which then can be solved. However, this method of creating expression trees is limited, since it is conceivably hard to turn most of the more interesting questions into a tree format. A more common approach to developing more generalizable mathematical reasoning abilities for computers is to use Recurrent Neural Networks. In these kinds of papers, the task of solving math questions is treated more like a machine translation/NLP task so that each problem is "translated" to a solution. For example, a study done by the Facebook AI team uses sequence-to-sequence models by feeding the NN a sequence of expressions representing math problems. Similar to the other study, this study mainly focuses on problems that can be converted into a tree structure, which are then converted to a sequence of expressions. Another study by David Saxton et al. similarly used LSTM and seq2seq models. However, they differed in their approach by inputting problems in free text format instead of preprocessing them into a structure. They generated their own dataset of problem-solution pairs in textual input/output form, and they compared the performances of the two models. 

In our research, we will try to implement the method that Saxton et al. applied in their paper and will use the synthetic dataset from their study in doing so. However, we will initially limit ourselves to trying to train a model to solve problems in only one field, which we might then apply to other fields. Through this research, we hope to get a better understanding of LSTM architectures by doing the implementation ourselves, and getting an intuition of why it might be better at solving some kinds of problems compared to other models. We think their method overcomes the difficulty of having to structurally represent problems in some format, and seems most open to experiment on. 

Some techincal challenges we anticipate facing include creating meaningful embeddings, deciding how to parse the input, and getting a sound understanding of how we can best utilize the LSTM architecture for our own problem. We also have to deal with limited computing resources, and we expect the size of our training dataset to potentially decrease the accuracy of our model. Minimally, we expect our model to perform with 70% accuracy for the arithmetic problems, which will be our first and primary focus.

After completing our research and training our model, we hope to be able to get a sense of what kinds of problems our model is best at solving. We might also compare the results of our study with those in Yanyan et al.'s paper that tried to create expression trees to solve arithmetic word problems. 

## Methods

As we mentioned earlier, we use the [dataset](https://github.com/deepmind/mathematics_dataset) containing millions of problem solution pairs for different math modules. We are using the `tensorflow` library for our NN models in addition to `scikit` for some helpful methods during training. We trained our model on one simple module (Arithmetic). Our model is a seq2seq neural network that is composed of the following layers

`Embedding -> Encoder LSTM -> RepeatVector -> Decoder LSTM -> Fully Connected -> Softmax` 

If we let `n,m` be the length of a given (problem, solution) pair in terms of number of characters in each of them, and let `v` be the number of unique characters in our dataset. The seq2seq model takes in a `n x 1` vector representing the characters in the problem(with indices according to a vocabulary), and outputs a `m x v` matrix where each row is represnting a probability distribution over the vocabulary indices for each character of the answer. Then, these rows representing probability distributions over the vocabularity for each character of the answer can be one-hot decoded to actual letters. 

To train the seq2seq model on batches, we had to specify a certain length both for the input sequence(question) and the output sequence(answer). So, we chose (20,5) and (30,10) as two pairs of maximum lengths for questions and answers, which meant that we only trained and tested our models on a subset of the original dataset. To make all questions and answers the same length, we added right-padding to complete the questions and answers to the specified length. This meant, for example, that we had quite many empty pad characters added to the end of many questions and answers. Essentially, this posed a challenge for us, since we had to "tell" the NN to ignore these as meaningless characters. This was achieved by adding the "mask_zero=True" option to the embedding layer. 

## Discussion

We tried several configurations of our model for this task. The base model we built was an LSTM layer connected to a Linear Layer that produced a single integer output.  We first tried this model to solve problems in the arithmetic module. In our first training setup, we used stochastic gradient descent with MSE as our loss function. The initial hyparameters and setup of our network is given as a table below

Accuracy

| Size of data set/ (n, m)   | (20,5) | (30,10) |
| :---:                      | :---:  | :---:   |  
| 1000                       |        |         |
| 5000                       |        |         |
| 10000                      |        |         |



| Hyperparameters                              |         Setup                          |
| :---:                                        | :---:                                  |
| X hidden states for LSTM,                    |    X optimization function             |
| X connected layers                           |    SGD                                 |
| 0.X learning rate                            |    MSE Loss                            |

This didn't give us desired results, so we tried training our model on different hyper parameters and optimization functions. The highest accuracy achieved in this module (X%) was obtained with the following combination of hyperparameters and setup: 

| Hyperparameters                              |         Setup                          |
| :---:                                        | :---:                                  |
| X hidden states for LSTM,                    |    X optimization function             |
| X connected layers                           |    X descent function                  |
| 0.X learning rate                            |    X Loss function                     |

These results are not surprising, since we expected accuracy to increase as we increased X,Y,Z and decreased X,Y,Z because we have seen in our initial configuration that X.  

Then, we tried our best model in other modules. For example, we got the lowest accuracy in the Probability module (X%). This might be becuase Probability questions exhibit X in contrast to arithmetic questions.  

When trained on all modules in the data set, our model performed with an overall accuracy of (X%). Overall, our model performed relatively well, and was slightly lower than Saxton et al. (X%). 

## Ethical Implications

The model that we will be training may be used for good purposes, such as checking automatically if a student got the right answer in a question. However, our peers whom we were doing peer reviews with also suggested the dual usage of this work, and questioned if this kind of module could also be abused by students for cheating. As our model currently is fairly simple and is only able to solve arithmetic problems, and only ones that do not have many words in them, we believe that our models do not currently pose such a threat. In addition, our models were not as accurate for validation tests, implying that the model's predictions are far from being reliable. However, thinking about the future of NN models capable of reasoning mathematically, this might be a valid concern. A more advanced NN model would be capable of solving math problems used for encryption. Hackers with access to such technology could potentially use it to decrypt private information.

An additional consideration is the language used in our data set. Our NN is evaluated on math problems that are written in English. Such an NN cannot be expected to perform as equally for problems that are in a different language. This means the technology developed from this model would not be equally accessible to people of all languages. Such a model would not be usable for a person who does math in a preferred or native language other than English.

Some of our peers also pointed out the question of what the philosophical implications of a mathematically reasoning computer would be for the field of math. So, if future work can find solutions to problems that are not easily solvable by humans, it might lead to further discussions about the nature of some problems. However, we believe that this is not much different from the question of whether complex NN models like the GPT can actually convince us that they are using languages as humans do. Overall, even though a computer might look like it has algebraic reasoning, we would still question the relevance of this kind of "reasoning" to our own human reasoning.

## Referenced Works

[**Text2Math: End-to-end Parsing Text into Math Expressions**](https://arxiv.org/abs/1910.06571)
(Yanyan Zou, Wei Lu. 2019)

[**Using neural networks to solve advanced mathematics equations**](https://ai.facebook.com/blog/using-neural-networks-to-solve-advanced-mathematics-equations/)
(Facebook AI Blog)

[**ANALYSING MATHEMATICAL REASONING ABILITIES OF NEURAL MODELS**](https://openreview.net/pdf?id=H1gR5iR5FX) (David Saxton et al.. 2019)

[**Solving Math Equations with Neural Networks**](https://ai.plainenglish.io/solving-math-equations-with-neural-networks-f015351995e8)(Blog post)
