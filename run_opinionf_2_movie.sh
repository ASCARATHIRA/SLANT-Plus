#!/bin/sh
python /home/ade/tf/temp/GTwitter/rnn_opinionf_2.py 567 90 ../567__Movie 100 > ../567__Movie/opf2_567_90_b100.log
python /home/ade/tf/temp/GTwitter/rnn_opinionf_2.py 567 90 ../567__Movie 500 > ../567__Movie/opf2_567_90_b500.log
python /home/ade/tf/temp/GTwitter/rnn_opinionf_2.py 567 90 ../567__Movie 1000 > ../567__Movie/opf2_567_90_b1000.log
python /home/ade/tf/temp/GTwitter/rnn_opinionf_2.py 567 90 ../567__Movie 5000 > ../567__Movie/opf2_567_90_b5000.log

