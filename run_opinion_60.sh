#!/bin/sh
python rnn_opinion.py 567 60 ../567__Movie > ../567__Movie/op_567_60.log
python rnn_opinion.py 548 60 ../548__Politics > ../548__Politics/op_548_60.log
python rnn_opinion.py 948 60 ../948__Fight > ../948__Fight/op_948_60.log
python rnn_opinion.py 1031 60 ../1031__Bollywood > ../1031__Bollywood/op_1031_60.log
python rnn_opinion.py 947 60 > op_947_60.log

