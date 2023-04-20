Randomness is the Root of All Evil: More Reliable Evaluation of Deep Active Learning (https://ieeexplore.ieee.org/document/10030143)

================================================
In this paper, we identify factors that influence the performance of different active learning strategies in three categories:
(1) the underlying learning setup,
(2) different sources of randomness, and
(3) specifics of the execution environment.

Methods under test. We re-implement 7 AL methods (BADGE [4 ], BALD [13], Core-Set [ 34], Entropy [ 35 ], ISAL [ 24], LC [ 23 ], and LLOSS [ 42 ])
across uncertainty-based, diversity-based, and combined strategies, and accompany them with a simple random selection strategy.

$\rightarrow$All experiments are conducted on the CIFAR-10 and CIFAR-100 dataset [20]

Each experiment is repeated multiple times and we randomly select 1,000 data points for the initial set and retrieve 2,000 samples within a query batch,
unless otherwise specified.

Moreover, we observe that BALD, BADGE, Entropy, ISAL, LLOSS and LC arrive at the maximum accuracy (the accuracy of a network trained on the entire training dataset) at 20 k to 30 k labeled samples.
For a conclusive evaluation, thus, comparing performance from zero to this converged point is crucial. Hence, in all subsequent experiments, we display results from 0 to 25 k labeled samples across different batches on the x-axis.

All experiments are conducted using NVIDIA A-100 GPUs, except for measurements performed to compare the influence of different hardware and non-deterministic training which are run on NVIDIA RTX 3090 cards.

Additionally, it is of the utmost importance that the community agrees upon a common definition of the individual backbones, so that, for instance, ResNet18 of a specific evaluation is identical with the used ResNet18 of another work.

Compared to stochastic gradient descent (SGD) [43], adaptive optimizers such as Adam [ 16] and RMSProp [12] show poor generalization despite faster convergence [40 , 44 ].
**Hence, the choice of optimizer can be crucial for assessing the final performance.** Beck et al. [5] and Lang et al.[22] conduct experiments to investigate the effect of using SGD and Adam optimizer, **concluding that SGD results in higher label efficiency with the same backbone$^1$ and hyper-parameters on the same dataset.** The significance of this is further underlined by the overview provided in Table 2, showing a diverse use of optimizers

$^1$backbone $\rightarrow$ "convolutional neural networks"

$\rightarrow$As SGD often generalizes better, we encourage its use for deep active learning
$\rightarrow$Pragmatically fix the learning rate to 0.1 for SGD on image datasets. While continuous hyperparameter tuning can improve overall performance, **a fixed learning rate does not change the ranking of AL methods from a comparative evaluation’s point of view.**

While data augmentation is popular in deep learning as a means to address overfitting, its significance for active learning is often neglected.
$\rightarrow$Beck et al. [5] show that overall accuracy and label efficiency can be improved with data augmentation.
$\rightarrow$Beside the overall improvement of classification performance, they also point out that data augmentation can affect the ranking of active-learning methods if used inconsistently. In contrast to adaptive hyper-parameter tuning data augmentation can be incorporated at comparably small training-time costs. Hence, a widespread use can be accepted more easily in practice.
$\rightarrow$One may use data augmentation if applied consistently across methods, such that it does not affect the overall ranking. However, a commonly accepted baseline is needed, e.g., random horizontal flipping and random cropping for image classification.

Yoo and Kweon [42] have identified 200 epochs as a practical setting for training ResNet18 on CIFAR-10, when the model is fully trained but not overfitting.
Using early stopping or a fixed number of epochs can have a impact on the evaluation.

Perhaps the most obvious influence factor on experimental design is randomness. While the need for controlling randomness is non-controversial [4, 25], the multitude of manifestations is difficult to oversee.
A common way of handling randomness in evaluations is to repeat each experiment several times and report averaged results with their standard deviation. However, if the fluctuation of a specific approach’s
performance is larger than the improvement over its contestants, results are difficult to interpret.

Key to controlling randomness is to update the backbone model as new samples are queried. Instead of learning the model from scratch with new random initialization, the model is initialized with the parameters from the previous round (“warm starts”). Consequently, all remaining randomness stems from the initialization in the first round of active learning, which already stabilizes the comparative
measurements significantly. For reliable comparative evaluation, initialization is fed with fixed inputs over multiple runs to average out the remaining randomness. This forms a sequence of R tuples, (s 1 , . . . , s l ) , containing seeds for initializing the individual factors for one specific run. As active learning is runtime expensive, the number of repetitions is usually relatively small in practice. **It thus is crucial that all methods under investigation receive the same tuple of random seeds to establish consistency across experiments.**
$\rightarrow$Refine model parameters across AL batch (“warm starts”) to prevent exhaustive reinitialization and feed initialization of the backbone model’s weights and the “init sets” with fixed inputs over multiple runs to average out the randomness. Moreover, use identical seeds for all methods under investigation.

In summary, we observe that
(a) active learning strategies expose vastly different degrees of change for varying seeds and
(b) Entropy, BALD, and BADGE seem more robust to varying initialization compared to Core-Set, LLOSS and ISAL.

While we are the first to discuss the use of “warm starts” for stabilizing active learning for evaluation purposes, the fact that it yields different performance than repeated “cold starts” has been investigated in prior work [5, 22 ]. In our experiments that we report in Appendix C, we confirm this in our setting as well.

Run experiments multiple times to compensate for non-deterministic operations. If the resulting variance is larger than the gained improvement, use deterministic operations more strictly.

$\rightarrow$Ensure that comparative evaluations are run on identical hardware. While it is not necessary to execute all experiments on the same physical device, the GPU model, for instance, should be the same.
Do not mix hardware and list hardware details.

$\rightarrow$Consider multiple query-batch sizes in the evaluation. The choice of the sizes needs to be appropriate for the total number of unlabeled samples.

$\rightarrow$Compare active learning strategies without sub-sampling, unless one of the approaches uses it as a fundamental building block. In this case a detailed analysis of the influence of sub-sampling is necessary.

the ranking of AL methods differs between balanced and imbalanced datasets.
$\rightarrow$Evaluate active learning strategies on multiple benchmark datasets, that comprise balanced, imbalanced, small-scale, and large-scale datasets to cover most relevant cases in practice.

==========================================
