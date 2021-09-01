#!/bin/bash


echo "Wish you get 0.9 f1 score this time..."

echo "Please enter project name, if you don't ,it will be exp"
read name

if [ -z "$name" ]; then
    name='exp'
fi

echo "Do you want to train model one by one? Input anything"
read trainsplit

if [ -z "$trainsplit" ]; then
    trainsplit=0
else
    trainsplit="one_by_one"
    echo "Do you want to split wandb project one by one too? Input anything"
    read projectsplit
fi
echo "Please enter model number from below or Input model name what you want."

echo "=============================================="
echo "===============Model Menu====================="
echo "            1. EfficientNetB3		            "
echo "	    2. Res2Next50		                    "
echo "	    3. ResNext50		  	                "
echo "	    4. DenseNet121			                "
echo "	    5. Inception-ResnetV2		            "
echo "	    6. Inception-ResnetV1,FaceNet           "
echo "	    7. I want another model                 "
echo "=============================================="

if [ -z "$trainsplit" ]; then
    read modelnumber
    case $modelnumber in
        1)
            modelname="EfficientNet_b3"
        ;;
        2)
            modelname="Res2Next50"
        ;;
        3)
            modelname="ResNext50"
        ;;
        4)
            modelname="DenseNet121"
        ;;
        5)
            modelname="InceptionResnetv2"
        ;;
        6)
            modelname="InR"
        ;;
        7)
            echo "Enter the name of model you want to apply"
            read modelname
        ;;
    esac
    echo "You chose $modelname model"
else
    n=0
    modelnames=""
    while [ ${n} -le 2 ]; do
        read modelnumber
        case $modelnumber in
            1)
                modelname="EfficientNet_b3"
            ;;
            2)
                modelname="Res2Next50"
            ;;
            3)
                modelname="ResNext50"
            ;;
            4)
                modelname="DenseNet121"
            ;;
            5)
                modelname="InceptionResnetv2"
            ;;
            6)
                modelname="InR"
            ;;
            7)
                echo "Enter the name of model you want to apply"
                read modelname
            ;;
        esac
        case $n in
            0)
                echo "You chose $modelname model for mask"
            ;;
            1)
                echo "You chose $modelname model for gender"
            ;;
            2)
                echo "You chose $modelname model for age"
            ;;
        esac
        n=$((n + 1))
        modelnames+="${modelname}"
        if [ ${n} -le 2 ]; then
            modelnames+=','
        fi
    done
fi

echo "Please enter epoch number, if you don't, it will be 10."

read epochnumber

if [ -z "$epochnumber" ]; then
    epochnumber=10
fi

echo "Choose Dataset, Default is 1."
echo "=============================================="
echo "=================Aug Menu====================="
echo "            1. MaskSplitByClassDataset		"
echo "	    2. MaskBaseDataset		                "
echo "	    3. I want another Dataset               "
echo "=============================================="
read dataset

if [ -z "$dataset" ]; then
    dataset=1
fi

case $dataset in
    1)
        dataset="MaskSplitByClassDataset"
    ;;

    2)
        dataset="MaskBaseDataset"
    ;;
    3)
        echo "Enter the name of dataset you want to apply"
        read dataset
    ;;
esac
echo "You chose $dataset for dataset"

echo "Choose Augmentations, Default is 1."
echo "=============================================="
echo "=================Aug Menu====================="
echo "            1. BaseAugmentation		        "
echo "	    2. CustomAugmentation		            "
echo "	    3. I want another Aug                   "
echo "=============================================="
read Aug

if [ -z "$Aug" ]; then
    Aug=1
fi

case $Aug in
    1)
        Aug="BaseAugmentation"
    ;;

    2)
        Aug="CustomAugmentation"
    ;;
    3)
        echo "Enter the name of Aug you want to apply"
        read Aug
    ;;
esac
echo "You chose $Aug for augmentations"

if [ -z $trainsplit ]; then
    python3 train.py --model $modelname --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug
elif [ -z "$projectsplit" ]; then
    python3 train.py --models $modelnames --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug --train_split $trainsplit
else
    python3 train.py --models $modelnames --epoch $epochnumber --dataset $dataset --name $name --augmentation $Aug --train_split $trainsplit --project_split $projectsplit
fi








