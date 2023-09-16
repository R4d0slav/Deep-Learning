# Building a Neural Network from scratch

## Introduction
This assignment is focused on implementing a simple neuron network with all its basic functionalities: forward propagation, backward propagation and network updating. Furthermore, several functions and approaches are explored, with different parameters, such as:

* Optimization functions:
*    Stocastig Gradient Descent (SGD)
*    ptive Moment estimation (Adam)
* Regularization
*   L2 regularization
* Learning rate schedule
*   Exponential decay

## Methodology
The network architecture is made up of N hidden layers, all containing sigmoid activation function, defined as: $\sigma (z) = \frac{1}{1+e^{−z}}$. The network is used for classification of 10 classes, thus the final output layer contains softmax activation function, defined as: $a_L^j = \frac{e^{z^L_j}}{\sum_k e^{z^L_k}}$. Finally, cross-entropy is used as loss function, which is defined as: $C = −\frac{1}{n}\sum_{xj} \[ y_j\ ln\ a^L_j + (1 −y_j)\ ln\ (1 −a^L_j ) \]$.

### Network implementation
For the forward propagation, for all layers, the corresponding outputs of the previous layers (or input for layer 1) are multiplied with the corresponding weights and then summed with the corresponding biases, using the following formula: $z_l = w_la_{l−1} + b_l$. After which, the results are passed through a sigmoid activation function for all layers except the last layer, where the results are passed through a softmax function. Furthermore, the results of each layer are saved for later calculations in back propagation.

For the back propagation, the calculations start from the change of the final output layer and the loss function and iteratively go all to the change from the input layer. For the change between the output layer and cross-entropy loss function, delta is calculated as simply the difference between the output and the target values $(y_i − \hat{y_i})$, the gradient for the biases is then simply the mean of delta along the rows $\frac{1}{n} \sum_{i=1}^n \delta \[i,:\]$, and the gradient for the weights is calculated as dot product between delta and the outputs of the previous layer, all divided by the batch size $\frac{1}{m} \delta \cdot (a_l−1)^T$. For the other layers, delta is calculated as the dot product of the weights in the next layers and previous delta, multiplied by the derivative of sigmoid for the output of the current layer. The gradients for biases and weights are calculated the same as before.

For the network update, the weights and biases of each layer are calculated and updated by either SGD or Adam, explained in later in more detail.

### Regularization
For the network, L2 regularization is implemented, simply by adding the regularization term $\frac{\lambda}{2n} \sum_w w^2$ to the loss function, encouraging the model to have small weights, thus limiting the models complexity which makes it less likely to overfit.

### Optimizer
For the optimizer function, two approaches are explored, SGD and Adam. In SGD the weights and biases are updated by simply subtracting the current values from the multiplication between the learning rate and the gradients for the corresponding weights and biases. Where as for Adam, which is a popular optimization algorithm that combines the ides from RMSprop and momentum, the weights and biases are updated using the following formulas:

$$ \sum \nabla = \beta_1 \sum \nabla + (1 - \beta_1) \nabla $$

$$ \sum^2 \nabla = \beta_2 \sum^2 \nabla + (1 - \beta_2) \nabla^2 $$

$$ \Delta = \frac{-\eta \sum \nabla}{\sqrt{\sum^2 \nabla} + \epsilon} $$


Where the decay rates, $\beta_1$ is usually 0.9 and $\beta_2$ is 0.999, $\eta$ is the learning rate, $\epsilon$ is a small constant to avoid division by zero, and the first and second formulas are the first and second moment estimates.

### Learning rate schedule
Finally, for the learning rate schedule, an exponential decay is used, slowly reducing the learning rate over time, using the following formula: $\eta_t = \eta e^{(−kt)}$, where $\eta$ is the original learning rate, k is the decay rate and t is the current epoch number. Usually at the beginning the learning rate is set a little higher, and then is gradually decreased over time, allowing the network to make finer adjustments

## Network training and results
The implemented neural network is trained on a variety of parameters:
* Number of layers and neurons - LN
* Learning rate - LR
* Number of epochs - Epochs
* Batch size - BS
* Regularization - Reg
* Decay rate - DR

When training a neural network, its best to first start simple, by using one hidden layer and slowly complicate it until it doesn’t make sense anymore. That is the approach used when training and testing this network for this assignment. Firstly, the network is evaluated when trained with just 1 hidden layer, no regularization and decay, tested on a variety of learning rate values, slowly complicating it by using several hidden layers, until the model becomes too complicated for the task and preforms poorly.

| Opt  | NL          | LR   | BS  | Epochs | Reg| DR  | Acc      |
|------|-------------|------|-----|--------|----|-----|----------|
| SGD  | 64          | 0.1  | 32  | 30     | L2 | 0   | 44.66%   |
| SGD  | 256         | 0.05 | 64  | 30     | L2 | 0   | 47.79%   |
| SGD  | 256         | 0.05 | 64  | 30     | /  | 0   |**48.25%**|
| SGD  | (128,64)    | 0.05 | 128 | 30     | L2 | 0.1 | 47.78%   |
| SGD  | (128,64)    | 0.05 | 128 | 30     | L2 | 0   | 44.92%   |
| SGD  | (128,128)   | 0.05 | 16  | 40     | L2 | 0.01|**49.51%**|
| SGD  | (256,128,64)| 0.01 | 64  | 30     | L2 | 0.01| 33.37%   |
| Adam | 128         |0.001 | 64  | 30     | /  | 0   | 46.77%   |
| Adam | 128         |0.001 | 64  | 30     | L2 | 0.01| 47.04%   |
| Adam | (128,64)    | 2e-5 | 128 | 30     | /  | 0.01| 37,47%   |
| Adam | (128,64)    | 2e-5 | 128 | 30     | /  | 0.01| 39.61%   |
| Adam | (256,128)   | 2e-4 | 16  | 30     | L2 | 0.01|**51.95%**|
| Adam | (256,128)   | 2e-3 | 16  | 50     | L2 |0.001| 37.87%   |


From the table above, we can see that by using 1-2 hidden layers, we get better results than by adding more hidden layers, further complicating it and slowing down training as well. For overall performance, on a more simple architecture, even on same parameters except learning rate, SGD outpreformed Adam. However, by using Adam with 2 hidden layers, (256, 128) and a low learning rate of 2e-4, with batch-size of 16 trained on 30 epochs using 0.01 decay rate and L2 regularization, we received highest accuracy of all other parameters, with 51 95%. Increasing the number of epochs to 50, even with L2 regularization, the model overfitted after epochs 30, where the validation loss started to increase and the accuracy started to decrease.

Furthermore, by omitting L2 regularization, slightly better results were received, for both Adam and SGD. However, without L2, on a large number of epochs, the models starts to overfit, where the validation loss start to increase. And also, by using a learning rate schedule, the performance of the model improved. Thus, a small decay rate was used, decreasing the learning rate at each epoch. And finally, most important parameter that seemed to substantially increase the accuracy was the batch size. By using a smaller batch size of 16, the model performed better then all other examples, as it updated the weights and biases more frequently. However, the drawback by using such low batch size is the computational cost, making training way to slow in comparison with higher batch sizes.

## Conclusion

In this assignment, a basic neural networks is implemented and tested with a variety of parameters. From the evaluation results, SGD by using a simple architecture on average outpreformed Adam, even when using the same parameters.
However, Adam gave the best overall performance, with accuracy of 51.95%. A few parameters were tested against each other, where omitting L2 regularization seem to improve accuracy on epochs around 30, however by increasing the number of epochs, the models started to overfit. Furthermore, by adding a learning rate schedule and also the using smaller batch sizes, the model preformed better,

The results could be improved by maybe increasing the complexity of the network by using Adam, as it seemed to perform better when more complicated than SGD. Also, the results could be improved by using different learning rate schedule types and increasing the number of epochs alongside other regularization functions or adding dropout layers and early-stopping when the model starts to overfit.

