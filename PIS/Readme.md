71% 성능을 보인 것은 working이지만 코드가 정리가 되어있지 않습니다.

Modified_working_with_load는 코드를 정리했고 학습 후 MyModel로 합친 것이 아닌 MyModel 자체를 학습시킵니다.

Modified_working_with_load가 epoch을 반복하면 memory 에러가 나서 한번 학습할 때마다 kernel을 restart해줘야했습니다.
그래서 train시 저장이 되게 하였고, 반복학습은 latest model을 불러와서 진행했습니다.

Modified_working_checkpoint는 반복학습이 가능한데 프로젝트 이름을 지정해줘서 체크포인트와 submission을 매 epoch마다 저장하고 kernel을 restart할 경우 최근 checkpoint로 모델이 초기화되게끔 했습니다. 5 epoch 학습시킨 예측결과는 이전보다 안좋아졌습니다;;