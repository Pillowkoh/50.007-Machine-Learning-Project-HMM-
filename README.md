# 50.007-Machine-Learning-Project-HMM

## Project Summary

Many start-up companies are interested in developing automated systems for analyzing sentiment information associated with social media data. Such sentiment information can be used for making important decisions such as making product recommendations, predicting social stance and forecasting financial market trends.

The idea behind sentiment analysis is to analyze the natural language texts typed, shared and read by users through services such as Twitter and Weibo and analyze such texts to infer the users’ sentiment information towards certain targets. Such social texts can be different from standard texts that appear, for example, on news articles. They are often very informal, and can be very noisy. It is very essential to build machine learning systems that can automatically analyze and comprehend the underlying sentiment information associated with such informal texts.

In this design project, we would like to design our sequence labelling model for informal texts using the hidden Markov model (HMM) that we have learned in class. We hope that our sequence labelling system for informal texts can serve as the very first step towards building a more complex, intelligent sentiment analysis system for social media text.

## Setting Up The Environment

Ensure that the project directory is setup in the following structure:

```
Datasets/
	ES/
		train
		dev.in
		dev.out
	RU/
		train
		dev.in
		dev.out
	ES-test/
		test.in
	RU-test/
		test.in
hmm.py
readfile.py
part4.py
main.py
```

## Running The Code

### How to use main.py

```
usage: main.py [-h] --part PART --datasets DATASETS [--k_num K_NUM] [--epochs EPOCHS]

optional arguments:

  -h, --help           	show this help message and exit
  
  --part PART          	Possible parts: 1, 2, 3, 4
  
  --datasets DATASETS  	Input datasets ES/RU/ES-test/RU-test. Separate by comma if there
  						are multiple input datasets. Ensure that datasets are stored in 
  						'Datasets/'
  						
  --k_num K_NUM        	Used when running part 3. Defines the k-th best sequence to
  						obtain. Defaults to 5.
  						
  --epochs EPOCHS      	Used when running part 4. Defaults to 10.
```



## Part 1

Using only emission probabilities to predict the output sequence.

Run the following command in the terminal:

```
python main.py --part=1 --datasets=ES,RU
```

**Output files:** /Datasets/{datasets}/dev.p1.out

&nbsp;

## Part 2

Using emission and transition probabilities, and Viterbi algorithm to predict the output sequence.

Run the following command in the terminal:

```
python main.py --part=2 --datasets=ES,RU
```

**Output files:** /Datasets/{datasets}/dev.p2.out

&nbsp;

## Part 3

Similar to Part 2, but Viterbi algorithm is modified to find the 5-th best output sequence.

Run the following command in the terminal:

```
python main.py --part=3 --datasets=ES,RU [--k_num=k]
```

**Output files:** /Datasets/{datasets}/dev.p3.out

&nbsp;

`--k_num` is used to determine the order of k-th best output sequence that the Viterbi algorithm will predict. 

The value of `--k_num` is defaulted to 5. Only specify this parameter if you wish to find out other orders of k-th best output sequence.

&nbsp;

## Part 4

Uses Structured Perceptron model for a discriminative approach of predicting the output sequence.

Choose one of the following commands to run in the terminal:

&nbsp;

### Getting predictions for dev.in files:

```
python main.py --part=4 --datasets=ES,RU [--epochs=n]
```

**Output files:** /Datasets/{datasets}/dev.p4.out

&nbsp;

### Getting predictions for test.in files:

```
python main.py --part=4 --datasets=ES-test,RU-test [--epochs=n]
```

**Output files:** /Datasets/{datasets}/test.p4.out

&nbsp;

`--epochs` is used to control the number of epochs to be iterated for the training process before predicting the test dataset.

The value of `--epoch` is defaulted to 10.

`--datasets` can take different arguments such as ES, RU, ES-test, RU-test.

