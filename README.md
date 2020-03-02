# Text prediction using LSTMs

Ever wanted to talk to some of the greatest writers in brazilian literature? You still can't, he's dead. But you can talk to my tekegram bot based on his texts, ~spoiler: it sucks~ .

### Note: if you have CUDA available change the `LSTM` occurrences to `CuDNNLSTM` for much faster training.

Ever wanted to talk to some of the greatest writers in Brazilian literature? You still can't, he's dead. But you can talk to my telegram bot based on his texts, ~~spoiler: it sucks~~.

To run the bot on your own server, just execute 

    python3 bot.py

And it will run with the architecture specified (the code already ships with trained weights). Make sure to have a valid access token in the code though. Check [this link] (https://core.telegram.org/bots) if you have any doubts.

To train the network with your own texts you fist need to preprocess it specifying the text file in `preprocessing.py` to remove noise, and then the preprocessed file in `model.py`.

Then just run:

    python3 model.py

Note that this performs a grid search to find the best network architecture and then trains it, so it might take some time.

## This work builds upon the one presented below, if you are interested please check [this blog post](https://medium.com/towards-artificial-intelligence/sentence-prediction-using-word-level-lstm-text-generator-language-modeling-using-rnn-a80c4cda5b40).


## Acknowledgements from the original author
* This project is highly based on this [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 
* Additional Readings: 
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [A Brief Summary of Maths Behind RNN ](https://medium.com/towards-artificial-intelligence/a-brief-summary-of-maths-behind-rnn-recurrent-neural-networks-b71bbc183ff)
* [How many LSTM cells should I use?
](https://datascience.stackexchange.com/questions/16350/how-many-lstm-cells-should-i-use/18049)
* [What's the difference between a bidirectional LSTM and an LSTM?
](https://stackoverflow.com/questions/43035827/whats-the-difference-between-a-bidirectional-lstm-and-an-lstm)
* [An Introduction to Dropout for Regularizing Deep Neural Networks](https://medium.com/towards-artificial-intelligence/an-introduction-to-dropout-for-regularizing-deep-neural-networks-4e0826c10395)
 one presented below, if you are interested please check [this blog post](https://medium.com/towards-artificial-intelligence/sentence-prediction-using-word-level-lstm-text-generator-language-modeling-using-rnn-a80c4cda5b40) .
