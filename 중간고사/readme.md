# 중간고사 과제

## 1. 수행 과정과 결과 정리

```python
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
```

사이킷 이미지를 활용하기 위한 모듈들을 import해주는 코드입니다.

- imread -> 이미지를 읽습니다.
- resize -> 이미지 크기를 변경합니다.
- hog -> 기울기와 히스토그램을 계산합니다.

```python
url = 'https://github.com/dknife/ML/raw/main/data/Proj2/faces/'
```

이미지 파일을 불러올 URL입니다.

```python
face_images = []

for i in range(15):
    file = url + 'img{0:02d}.jpg'.format(i+1)
    img = imread(file)
    img = resize(img, (64,64))
    face_images.append(img)


plot_images(3,5, face_images)
```

이미지 파일명을 만들고 이미지 크기를 조정한 후 배열에 추가합니다.

```python
face_hogs = []
face_features = []

for i in range(15):
    hog_desc, hog_image = hog(face_images[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    face_hogs.append(hog_image)
    face_features.append(hog_desc)

plot_images(3, 5, face_hogs)

print(face_features[0].shape)
```

얼굴 이미지의 특징 데이터를 구해 저장하는 로직입니다.

기울기 히스토그램을 사용해 이미지를 2차원 공간을 정의역으로 하고 픽셀 값을 치역으로 해 기울기를 계산합니다. 이 기울기를 8개 방향으로 구분하여 각 방향별 빈도를 계싼하면 기울기의 히스토그램을 구할 수 있습니다. 이런 동작을 위 hog 함수가 수행합니다.

이후 히스토그램 이미지를 가시화하면 16개의 블록과 16개의 히스그램이 그려질 것입니다.

```python
fig = plt.figure()
fig, ax = plt.subplots(3,5, figsize = (10,6))
for i in range(3):
    for j in range(5):
        ax[i, j].imshow(resize(face_features[i*5+j], (128,16)))
```

1차원 벡터이면서, 128x1의 이미지를 눈으로 확인하기 쉽게 128x16 크기로 변경해 확인해보는 로직이며, 이 벡터가 각 이미지의 특징 벡터라고 할 수 있습니다.

#### 지금까지는 정 그룹의 데이터, 즉 사람 얼굴 데이터를 저장했으며, 다음으로는 부 그룹의 데이터가 될 이미지들을 준비합니다.

**코드 및 로직은 이미지 파일을 가져오는 경로만 다르기에 추가 설명을 따로 추가하지 않았습니다.**

```python
url = 'https://github.com/dknife/ML/raw/main/data/Proj2/animals/'

animal_images = []

for i in range(15):
    file = url + 'img{0:02d}.jpg'.format(i+1)
    img = imread(file)
    img = resize(img, (64,64))
    animal_images.append(img)

plot_images(3, 5, animal_images)

animal_hogs = []
animal_features = []

for i in range(15):
    hog_desc, hog_image = hog(animal_images[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    animal_hogs.append(hog_image)
    animal_features.append(hog_desc)

plot_images(3, 5, animal_hogs)

fig = plt.figure()
fig, ax = plt.subplots(3,5, figsize = (10,6))
for i in range(3):
  for j in range(5):
    ax[i, j].imshow(resize(animal_features[i*5+j], (128,16)))
```

이제 정, 부 데이터 그룹을 생성했고 입력에 대해 얼굴은 1, 사람 얼굴이 아닌 입력에는 0을 입력하는 배열을 준비합니다.

```python
X, y = [], []

for feature in face_features:
    X.append(feature)
    y.append(1)
for feature in animal_features:
    X.append(feature)
    y.append(0)

fig = plt.figure()
fig, ax = plt.subplots(6,5, figsize = (10,6))
for i in range(6):
  for j in range(5):
    ax[i, j].imshow(resize(X[i*5+j], (128,16)),interpolation='nearest')
print(y)
```

이제 모든 데이터가 준비됐으니 학습을 시켜보도록 하겠습니다.

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

polynomial_svm_clf = Pipeline([("scaler", StandardScaler()), ("svm_clf", SVC(C=1, kernel = 'poly', degree=5, coef0=10.0))])
polynomial_svm_clf.fit(X, y)

Pipeline(memory=None, steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm_clf', SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=10.0, decision_function_shape='ovr', degree=5, gamma='scale', kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False))], verbose=False)

yhat = polynomial_svm_clf.predict(X)
print(yhat)
```

svm을 활용해 학습을 시킨 후 데이터를 출력해보면 다음과 같은 결과를 확인할 수 있습니다.

<img width="501" alt="스크린샷 2022-10-30 오후 5 15 35" src="https://user-images.githubusercontent.com/78959175/198869030-2d6ba59a-717a-44d2-a864-ea5cdc4b1fe1.png">

정상적인 결과가 나옵니다.

## 2. 이미지 중 일부를 자신의 사진으로 변경 후 결과 정리

마지막으로 이미지 중 일부를 제 사진으로 변경 후 테스트 해보겠습니다.

초기 이미지 파일들은 제공해주는 파일들로 테스트 했으며, **img02.jpg** 파일과 **img08.jpg** 파일은 제 사진으로 대체해서 진행했습니다.

<img width="228" alt="스크린샷 2022-10-30 오후 4 55 35" src="https://user-images.githubusercontent.com/78959175/198869191-8d1b4aaf-c323-417b-bb97-1acbc5cc32a6.png">

이미지를 대체하기 전 결과입니다. 여기서 주목하실 사람 얼굴 데이터는 2번째 요소와 8번째 요소입니다.

#### 이미지 대체 후 결과

<img width="230" alt="스크린샷 2022-10-30 오후 4 55 40" src="https://user-images.githubusercontent.com/78959175/198869214-a5079a5e-2ee2-4ade-8198-4b8d91de2370.png">

해당 이미지 결과를 확인해보면 8번째 요소는 제대로 얼굴을 인식하나 2번째 요소는 그대로 사람 얼굴인지 인식하지 못합니다.

그 이유는 해당 이미지에 있습니다.

우선 8번째로 대체된 이미지 파일입니다.

![img08](https://user-images.githubusercontent.com/78959175/198869268-8878f830-f0bb-4aa3-8028-66be17b7ca68.jpg)

해당 이미지는 증명 사진을 사용했기 때문에 정상적으로 인식합니다.

다음으로는 2번째로 대체된 이미지 파일입니다.

![img02](https://user-images.githubusercontent.com/78959175/198869295-35071aa0-5e7f-47a2-8e87-421406d6e88a.jpg)

해당 이미지는 마스크를 착용한 사진으로서 얼굴의 특징을 제대로 표현하지 못하기 때문에 프로그램 결과에서 얼굴로 인식하지 못한 것을 확인할 수 있습니다.

이상으로 중간고사 과제를 마치겠습니다.

**감사합니다.**
