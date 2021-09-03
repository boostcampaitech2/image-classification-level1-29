# Project : Mask Image Classification

주어진 사람 이미지를 3가지로 분류하는 문제입니다.



<br>

#### 3가지 기준

* **Mask** : `Not wear`,   ` Incorrect`,    `Wear`

* **Gender** : `Male`,    `Female`

* **Age** : `0 ~ 29`,    `30 ~ 59`,    `60~`



<br>



### Model Structure

![Untitled Diagram drawio (2)](https://user-images.githubusercontent.com/88299729/132023691-34ebd2b1-b857-4ef1-b7cf-633d259f14e7.png)


<br>



### 프로그램 개요

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.



<br>



### Requirements

필요한 패키지들은 아래 명령어를 통해 설치 가능합니다.



``` pip install -r requirements.txt 
pip install -r requirements.txt
```



<br>



### Get Started

#### 1. Source code clone 하기 

 ```
git clone https://github.com/boostcampaitech2/image-classification-level1-29.git
 ```



#### 2. data set 준비하기



```
+-- train/
l       +--images/
l              +--000001_female_Asian_45/
l              +--000002_male_Asian_52/
l              +--....
l       +--train.csv/
+-- eval/
l       +--images/
l              +--f4jk2h4jk35j3k2h5jk3.jpg/
l              +--g6hjk456jk5g6h45jkh.jpg/
l              +--....
l       +--info.csv/
```



#### 3. hyperparameter tuning을 하며 학습시키기



```
python train.py
--batch_size=32
--resize=(384,384)
--model=EfficientNet_b3
--nworkers=4
--lr=1e-3
--epochs=100
--optimizer=Adam
--criterion=cross_entropy
--scheduler=CosineAnnealingLR
--augmentation=CustomAugmentation
--patience=3
```