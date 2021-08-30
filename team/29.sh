#!/bin/bash


echo "Wish you 0.9 f1 score this time..."
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


case $modelnumber in
    1)
        python3 train.py --model EfficientNet_b3
    ;;

	2) 
        python3 train.py --model Res2Next50
    ;;

    3) 
        python3 train.py --model ResNext50
    ;;

    4) 
        python3 train.py --model DenseNet121
    ;;

    5)
        python3 train.py --model InceptionResnetv2
    ;;
    
esac








