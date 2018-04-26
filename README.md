# Domain-Adaptation-With-SNES
A Domain Adaptation approach, that uses an Evalutionary algorithm to train a Convalutional neural network to extract features that are transferable from source to target domain. 

This work was inspired by [Unsupervised domain adaptation by backpropagation](http://jmlr.org/papers/volume17/15-239/15-239.pdf).An evalutionary learning strategy is used as an alternative to the backpropagation and  transfer gradient reversal   used in the original work. domain_adaptation.py contains a python implementation of the appraoch, mnist_adaptation.py contains an expriments with MNIST data used as source domain and artificial dataset MNISTM created from MNIST is used as target data. office-31 adaptation contains an expriment  on the office dataset with the amazon dataset used as source data and the dslr dataset used as target.

 Method | Source_acc (MNIST) | Target_acc(MNISTM) (this repo w/25 gen of SSNES) |
| ------ | ------------------ | ----------------------------------- |
| Source Only | 0.892 | 0.127 |
| Domain Adaptation | 0.891 | 0.123 |

 Method | Source_acc (amazon) | Target_acc(dlsr) (this repo w/25 gen of SSNES) |
| ------ | ------------------ | ----------------------------------- |
| Source Only | 1.00 | 0.022 |
| Domain Adaptation | 0.956 | 0.055 |
