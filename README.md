## Language-Modeling-using-RNNs
#Character and word level language models using LSTM


Dataset used: Penn Tree Bank



Char level RNN code has implementation for hyperparameter tuning using Tensorboard.
It was implemented using 1 LSTM layer and 1 densely connected layer.

For the word level RNN, we first split the training data into separate words, 
then each unique word is mapped to a unique integer and finally the original text file is converted into a list of 
unique integers, where each word is substituted with its new integer identifier. 
We then create n-gram sequences of maxlen 30 using this integer based file and call this 2-dim array of size 
(no-of- sequences, maxlen) as the predictors array. Simultaneously we prepare an array of labels containing 
the corresponding targets: one-hot-encoded characters that come after each extracted sequence.
We pass our predictors and labels into an embedding layer with 500 hidden units. 
The rest of the network is a double LSTM layer with 500 hidden units followed by a dropout layer and finally a 
Dense softmax classifier over all possible characters.


To Run(for both): 
Download the dataset and change the dataset file path location in the code file.


Results and graphs are summarized in the Report.pdf file.
