# deep-learning-sentiment-analysis
deep-learning-sentiment-analysis is an NLP project that compares three different models for binary sentiment classification. 

## Data
deep-learning-sentiment-analysis uses Stanford's [Large Movie Review Dataset] (http://ai.stanford.edu/~amaas/data/sentiment/). This dataset was designed for NLP sentiment analysis and published by [Maas et al] (http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). It consists of 12,500 positive train, negative train, positive test, and negative test reviews, along with 50,000 unlabeled reviews for unsupervised learning, for 100,000 total reviews.

## Models
deep-learning-sentiment-analysis utilizes three different models for sentiment analysis:
* Recursive Neural Tensor Network via [Stanford CoreNLP] (http://nlp.stanford.edu/sentiment/code.html)
* Doc2Vec embedding via [gensim] (https://radimrehurek.com/gensim/models/doc2vec.html)
* Convolutional Neural network via [TensorFlow] (https://www.tensorflow.org/)

## Academic Background
* Dataset: [Maas et al.] (http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
* Recursive Neural Tensor Network: [Socher et al. 2013] (nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
* Word2Vec: [Mikolov et al. 2013] (https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
* Doc2Vec: [Le and Mikolov 2014] (https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
* Convolutional Neural Network: [Kim] (https://arxiv.org/pdf/1408.5882v2.pdf)

## Software Dependencies
deep-learning-sentiment-analysis is written in Python 2.7 in a Jupyter notebook and uses several common software libraries, most notably gensim and TFLearn. In order to run it, you  must install the follow dependencies:
* [Python] (https://www.python.org/)
* [Jupyter] (http://jupyter.org/)
* [gensim] (https://radimrehurek.com/gensim/)
* [TFLearn] (http://tflearn.org/)
* [NumPy] (http://www.numpy.org/)
* [BeautifulSoup] (https://www.crummy.com/software/BeautifulSoup/)

## License
This project uses the [Apache 2.0 License] (https://github.com/charlescc9/deep-learning-sentiment-analysis/blob/master/LICENSE).
