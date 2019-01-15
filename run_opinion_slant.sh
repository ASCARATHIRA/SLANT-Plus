#!/bin/sh
#python rnn_opinion_slant.py 567 70 ../567__Movie > ../567__Movie/op_567_70.log
python /home/ade/tf/temp/GTwitter/rnn_opinion_slant.py 548 90 ../548__Politics > ../548__Politics/opslant_548_90.log
python /home/ade/tf/temp/GTwitter/rnn_opinion_slant.py 948 90 ../948__Fight > ../948__Fight/opslant_948_90.log
python /home/ade/tf/temp/GTwitter/rnn_opinion_slant.py 1031 90 ../1031__Bollywood > ../1031__Bollywood/opslant_1031_90.log
#python rnn_opinion.py 947 70 > op_947_70.log

