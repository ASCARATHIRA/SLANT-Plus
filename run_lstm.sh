#!/bin/sh
python rnn_lstm.py 947 > lstm947.log
python rnn_lstm.py 567 ../567__Movie > ../567__Movie/lstm567.log
python rnn_lstm.py 548 ../548__Politics > ../548__Politics/lstm548.log
python rnn_lstm.py 948 ../948__Fight > ../948__Fight/lstm948.log

