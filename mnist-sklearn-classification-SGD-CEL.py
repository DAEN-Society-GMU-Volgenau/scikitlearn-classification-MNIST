"""

Recognize hand-written digits using Stochastic Gradient Descent

Darron Fuller - November 2015 - Society of Data Analytics Engineers at Volgenau School of Engineering
Web Page: www.DAEN-Society.org
eMail:  contact@DAEN-Society.org
Twitter:  @da_engineers
"""
print(__doc__)

# Standard scientific Python imports
import input_data   # borrowed this from TensorFlow example in order to load the same image data set as that example
import time

# Import datasets, classifiers and performance metrics
from sklearn import datasets, linear_model, metrics

# The digits dataset
digits = datasets.load_digits()

mnist = input_data.read_data_sets("MNIST_data/", one_hot = False)

print 'TRAINING data shape (cases,image dimension) : {:s}'.format(mnist.train.images.shape)
print 'TEST data shape (cases,image dimension) : {:s}\n'.format(mnist.test.images.shape)

# begin timing now (after loading data)
timeStart = time.clock()

# create Stochastic Gradient Descent classifier using Log loss function (equivalent to Cross-Entropy Loss function)
classifier = linear_model.SGDClassifier(loss="log",verbose=0)

# We learn the digits on the first half of the digits
classifier.fit(mnist.train.images, mnist.train.labels)

# Now predict the value of the digit on the second half:
expected = mnist.test.labels
predicted = classifier.predict(mnist.test.images)

# stop timer before report generation
timeElapsed = (time.clock() - timeStart)
print 'Run Time: %2.2f seconds.\n' % timeElapsed

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))




