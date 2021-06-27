# Capstone Design
------------
#### 주제
'''
Cut-Out을 이용한 Bias/Variance trade-off에 따른 data augmentation의 효과 검증
'''
------------
#### 연구 환경
'''
OS : Window 10, macOS M1(use Colab)
Code editor : Visual Studio Code 1.56.2
Python : 3.9.4
Pytorch : 1.8.1
Cuda : 11.1
Torchvision : 0.9.1
'''
------------
#### 코드
- train_sampling.py
	- cutout된 이미지로 학습한 8개의 모델을 생성
'''
python train_sampling.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
'''
- avg_var.py
	- train_sampling에서 생성된 checkpoint를 불러와 이미지를 저장하고 각 데이터의 variance를 계산
'''
python avg_var.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
'''
- test_n%.py
	- 상위 n% 데이터 제거 후 학습해 test_acc를 확인
'''
python test_n%.py --dataset cifar10
'''
#### 가설 설정
- 가설
'''
1. bias와 variance는 trade-off관계이고 현대 기계학습은 variance를 줄이는 것을 더 중요시한다.
2. data augmentation을 통해 얻은 새로운 데이터 중 variance가 큰 값을 학습에 사용하면 모델의 성능이 저하될 것이다.
--> variance가 큰 데이터를 제거하여 학습에 사용하면 augmentation만 적용하여 학습했을 때 보다 좋은 성능을 낼 것이다.
'''
------------
#### 연구 과정
- 전체 연구 과정
![제목 없음1](https://user-images.githubusercontent.com/74352090/123544696-6a617880-d78f-11eb-9b19-650cde3cbce9.png)

- 데이터 정렬 및 상위 n% 제거
![제목 없음2](https://user-images.githubusercontent.com/74352090/123544991-e8724f00-d790-11eb-8752-a68fe9ae8906.png)
------------
#### 문제점
- 마지막 단계 test에서 test_acc가 0.1로 고정
![제목 없음3](https://user-images.githubusercontent.com/74352090/123545169-b31a3100-d791-11eb-96e7-a4780a5669dd.png)
	- cifar-10은 10개 classes로 구성되어 있으므로, 10개 중 하나를 찍는 것과 동일한 결과
	- label을 확인해보니 training은 잘 되었지만 test에서 문제가 발생.
![제목 없음4](https://user-images.githubusercontent.com/74352090/123545207-e8268380-d791-11eb-8004-5270cf9975f9.png)
------------
#### 문제 분석
'''
1. 학습된 정보를 test에서 받지 못한다.
2. cifar-10에서 제공하는 데이터는 batch 형태로 제공되는 반면, custom dataset은 data와 label을 직접 불러온다.
3. 따라서 custom set과 test set의 구조가 다른 문제로 인해 test가 제대로 되지 않는 문제가 발생한 것 같다.
'''
------------
#### 결론
'''
연구 마지막 단계에서 학습된 정보가 test에 반영되지 않는 문제로 인해 가설을 검증하지 못했다.
비록 이번 연구에서 가설을 검증하지 못했지만, variance가 큰 데이터는 모델을 overfitting 시켜 일반화 성능이 저하되므로 variance가 큰 값을 제거하면 성능이 좋아질 것이다.
가설을 검증하기 위해 test set도 custom set과 같이 직접 저장해 dataset을 같은 구조로 만들어서 연구를 진행해본다.
'''