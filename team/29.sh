#!/bin/bash


echo "Wish you get 0.9 f1 score this time..."
echo "Please enter model number from below"

echo "=============================================="
echo "===============Model Menu====================="
echo "            1. EfficientNetB3		                "
echo "	    2. Res2Next50		                    "
echo "	    3. ResNext50		  	                "
echo "	    4. DenseNet121			                "
echo "	    5. Inception-ResnetV2		            "
echo "=============================================="

read modelnumber
echo "Please enter epoch number, if you don't, it will be 10."

read epochnumber

if [ -z "$epochnumber" ]; then
    epochnumber=10
fi

echo "0 to use Maskbaseset,1 to use MaskSplitDataset"
read dataset

if [ -z "$dataset"]; then
    dataset=1
fi

case $modelnumber in
    0)
        dataset="MaskBaseDataset"
    ;;

    1)
        dataset="MaskSplitDataset"
    ;;
esac




case $modelnumber in
    1)
        python3 train.py --model EfficientNet_b3 --epoch $epochnumber --dataset $dataset
    ;;

	2) 
        python3 train.py --model Res2Next50 --epoch $epochnumber --dataset $dataset
    ;;

    3) 
        python3 train.py --model ResNext50 --epoch $epochnumber --dataset $dataset
    ;;

    4) 
        python3 train.py --model DenseNet121 --epoch $epochnumber --dataset $dataset
    ;;

    5)
        python3 train.py --model InceptionResnetv2 --epoch $epochnumber --dataset $dataset
    ;;
    
esac








