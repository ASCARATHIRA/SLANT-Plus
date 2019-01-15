rnn_* codes are different models.
through_time_* codes do forecasting.

utility.py considers limited history - 15k messages. 
utility2.py considers all messages in a dataset. 
To use utility2, we have to use batched training like rnn_stacked_3f.py