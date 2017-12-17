# Decrypto Networko
In this project we aim to create a dynamic encryption algorithm using neural networks.<br>
Current encryption algorithms (even public key crypto algorithms) face a problem where the key is static in nature and needs to be changed manually in case of an attack. This can be changed with the help of neural networks. We aim to provide a solution where the neural network architecture and the weight matrices in it will act as the key.<br>
Thus our architecture not only makes it a 2 step process to get the key, but also makes it extremely hard to find out the key used.

## Libraries used
We will primarily be working with [keras](https://keras.io/) with a backend of [TensorFlow](https://www.tensorflow.org/) or [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) depending on the architecture. It has been seen that recurrent neural networks are much faster in CNTK. In case the architecture used is a type of recurrent neural network, then it is advised to use a CNTK backend instead of a tensorflow one.<br>

It is recommended to use [Anaconda](https://www.anaconda.com/download/) python as the default python environment as it solves a lot of dependency problems associated with the network.