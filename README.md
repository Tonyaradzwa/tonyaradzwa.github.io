# Title: Generating Hard Math Problems

![](images/maze-example.png)

## Project Description

I will focus on training neural networks (NNs) for the task of navigating a simulated "maze". A robot will be placed in a simulated environment (similar to the image above) and the NN will direct the robot to follow turn-by-turn directions from a starting location to an ending location. The simulator already exists, but a new dataset will need to be created, either manually or using an automated process.

The trained NN will be "deployed" as an application by putting the NN in new, unseen environments and testing its performance. If all goes well, then an extension of this work will include changing the simulation textures and lighting and training the NN to work when facing unseen textures.

The longer-term goal is to use a NN to enable better navigation in complex environments for a robot with multiple modes of locomotion (i.e., wheeled and flight). The results from this work will feed into an on-going project where we are trying to "cross the reality gap" that exists between simulation and the real world.

## Project Goals

1. Create a dataset for training a NN for navigation.
2. Explore methods for adding "noise" to the dataset.
3. Train a NN that is able to navigate any procedurally generated maze.
