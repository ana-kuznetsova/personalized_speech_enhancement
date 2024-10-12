#!/bin/bash

#speakers=(0 1 2)[]
speakers=(0 1 2 3 4)
sizes=('large' 'medium' 'small' 'tiny')
#sizes=('tiny')
rates=(0.000001)
#rates=(0.00001)


for spk in ${speakers[@]}; do
    for s in ${sizes[@]}; do
        for r in ${rates[@]}; do
           CUDA_VISIBLE_DEVICES=0 python my_run.py  -s $spk -r $r -i $s -p 'yourtts_30sec_10spk'
        done
    done
done