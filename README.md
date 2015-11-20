# scikitlearn-classification-MNIST

Demo that performs the same Stochastic Gradient Descent (SGD) classification on the MINST digit image data set as was performed in the TensorFlow demo (see TensorFlowDemo repository).  Timings are provided to demonstrate the efficiency of TensorFlow as compared to scikit-learn.

Planned Improvements:
1.  Online/Out-of-Core Learning.  SGD training on large data sets is an expensive operation.  Apply methods to incrementally train by partitioning the training data into smaller batches, training on each individual "mini-batch" of training data, and aggregate the results.
2.  Parallelization.  Out-of-Core learning (see #1) should be parallelizable to take advantage of multi-core systems.

These improvements should help to reduce processing time to that of TensorFlow (which employs both of these optimizations), but the additional code and complexity requried when using scikit-learn will further demonstrate the simplicity and power of the TensorFlow architecture.
