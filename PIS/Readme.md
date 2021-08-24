71% 성능을 보인 것은 working이지만 코드가 정리가 되어있지 않습니다.

Modified_working_with_load는 코드를 정리했고 학습 후 MyModel로 합친 것이 아닌 MyModel 자체를 학습시킵니다.

Modified_working_with_load가 epoch을 반복하면 memory 에러가 나서 한번 학습할 때마다 kernel을 restart해줘야했습니다.
그래서 train시 저장이 되게 하였고, 반복학습은 latest model을 불러와서 진행했습니다.