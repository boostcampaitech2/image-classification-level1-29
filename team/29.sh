#!/bin/bash


echo "Wish you get 0.9 f1 score this time..."

echo "Please enter project name, if you don't ,it will be exp"
read name

if [ -z "$name" ]; then
    name='exp'
fi

echo "Please enter model number from below"

echo "=============================================="
echo "===============Model Menu====================="
echo "            1. EfficientNetB3		                "
echo "	    2. Res2Next50		                    "
echo "	    3. ResNext50		  	                "
echo "	    4. DenseNet121			                "
echo "	    5. Inception-ResnetV2		            "
echo "	    6. Inception-ResnetV1,FaceNet           "
echo "=============================================="





read modelnumber
echo "Please enter epoch number, if you don't, it will be 10."

read epochnumber

if [ -z "$epochnumber" ]; then
    epochnumber=10
fi

echo "0 to use Maskbaseset,1 to use MaskSplitByProfileDataset"
read dataset

if [ -z "$dataset" ]; then
    dataset=1
fi

case $dataset in
    0)
        dataset="MaskBaseDataset"
    ;;

    1)
        dataset="MaskSplitByProfileDataset"
    ;;
esac
echo "choose augmentation, 0 to base, 1 to custom"
read Aug

if [ -z "$Aug" ]; then
    Aug=0
fi

case $Aug in
    0)
        Aug="BaseAugmentation"
    ;;

    1)
        Aug="CustomAugmentation"
    ;;
esac

case $modelnumber in
    1)
        python3 train.py --model EfficientNet_b3 --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug
    ;;

	2) 
        python3 train.py --model Res2Next50 --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug
    ;;

    3) 
        python3 train.py --model ResNext50 --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug
    ;;

    4) 
        python3 train.py --model DenseNet121 --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug
    ;;

    5)
        python3 train.py --model InceptionResnetv2 --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug
    ;;

    6)
        python3 train.py --model InR --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug
    ;;
    
esac








