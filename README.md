# GraduateProject : Efficient Face Detection Alg for low-end PC using OpenCV

## 1. 프로젝트 개요
 본 프로젝트는 카메라와 pc의 성능이 좋으면 높은 fps가 나와 효율적인 Face detection이 가능하지만 노트북의 웹캠, 모바일 등의 저사양 카메라 및 pc에서도 효율적인 Face detection을 하기 위해 기존 방법 보다 fps를 향상 시키는 것을 목적으로 한다. 기존에 보편적으로 사용되는 OpenCV를 활용하여 얼굴인식 작업을 수행하고 사용자가 이동함에 따라 동영상 배경이 달라지고 시선, 원근감에 따라 배경이 달라지는 Smart Window라는 SW를 제작하고 이후 처리 과정에서 최적화를 통해 fps를 향상시켰다. 기존 얼굴 인식 함수(face_classifier.detectMultiScale)는 단일 사용자를 대상으로도 사용자와 카메라 사이의 거리가 관계없이 고정된 크기의 minSize를 사용하여 불필요한 연산까지 하고 있는 것으로 보인다. 이 점을 활용하여 사용자와 카메라 사이의 거리가 가까운 경우 minSize값을 조절하는 것으로 연산량을 감소시켜 fps를 증가시킬 수 있었다. 이것은 저사양 pc에서의 Face detection을 통한 fps를 최대한 높여 요구되는 fps가 동일한 Face detection system을 활용하는 pc의 성능을 낮출 수 있을 것으로 보인다.
## 2. 프로젝트 목표
 기존 상용화 된 스마트 윈도우는 사람을 인식하기 위해 적외선을 이용하는 TOF 센서를 소지해야 하는 등의 제약이 있었다. 이것을 해결하기 위해 다른 보조장치 없이 동작하는 스마트 윈도우를 구현하고, 이 스마트 윈도우에 사용되는 pc와 카메라가 저사양의 pc 및 카메라에서의 Face Detection system의 효율을 최대한으로 끌어올리는 것을 최종목표로 한다. 이를 위해 저사양 노트북 및 웹캠 1대와 고사양 노트북 및 웹캠 1대를 준비하여 각각 코드 수정 이전의 fps와 저사양 노트북의 Face Detection system을 코드 최적화 이후의 fps와 고사양 노트북의 fps 향상 정도를 비교한다. 
## 3. 기초 기술
### I. Image Processing
영상 처리(Image Processing) 또는 화상 처리는 넓게는 입출력이 영상인 모든 	형태의 정보 처리를 가리키며, 사진이나 동영상을 처리하는 것이 대표적인 		예이다. 대부분의 영상 처리 기법은 2원 신호를 보고 여기에 표준적인 신호 		처리 기법을 적용하는 방법을 쓴다.  
20세기 중반까지 영상 처리는 아날로그로 이루어 졌으며 대부분 광학과 		연관된 방법이었다. 이런 영상 처리는 현재까지도 홀로그래피 등에 사용되지만, 	컴퓨터 처리 속도의 향상으로 인해 이런 기법들은 디지털 영상 처리 기법으로 		많이 대체되었다. 일반적으로 디지털 영상 처리는 다양한 방법으로 쓰일 수  		있으며 정확하다는 장점이 있고, 아날로그보다 구현하기도 쉽다.
### II. computer Vision
  컴퓨터 비전(Computer vision)은 기계의 시각에 해당하는 부분을 연구하는 		컴퓨터 과학의 최신 연구 분야 중 하나이다. 공학적인 관점에서, 컴퓨터 비전은 	인간의 시각이 할 수 있는 몇 가지 일을 수행하는 자율적인 시스템을 만드는 		것을 목표로 한다.
### III. Face Detection
  얼굴 검출(face detection)은 컴퓨터 비전의 한 분야로 영상에서 얼굴이 		존재하는 위치를 알려주는 기술이다. 얼굴 검출의 알고리즘적인 기본 구조는 		Rowley, Baluja 그리고 Kanade의 논문에 의해 정의되었다. 다양한 크기의 		얼굴을 검출하기 위해 피라미드 영상을 생성한 후, 한 픽셀 씩 이동하여 특정 		크기의 해당 영역이 얼굴인지 아닌지를 분류기(신경망), 아다부스트(Adaboost), 		서포트 벡터 머신(Support Vector Machine)로 얼굴인지 아닌지를 결정한다. 
### IV. OpenCV
  OpenCV(Open Source Computer Vision)은 실시간 컴퓨터 비전을 목적으로 		한 프로그래밍 라이브러리이다. 원래는 인텔이 개발하였다. 실시간 이미지 		프로세싱에 중점을 둔 라이브러리로 인텔 CPU에서 사용되는 경우 속도의 		향상을 볼 수 있는 IPP(Intel Performance Primitives)를 지원한다. 이 			라이브러리는 윈도우, 리눅스 등에서 사용이 가능한 크로스 플랫폼이며 			오픈소스 BSD 허가서 하에 무료로 사용할 수 있다. 또한 OpenCV는 			TensorFlow, Torch/PyTorch 및 Caffe의 딥러닝 프레임워크를 지원한다.
## 4. 결과 및 기대효과
  테스트는 저사양 노트북과 고사양 노트북 각각 코드 수정 이전과 이후,  근거리(노트북 캠에서 20cm)와 원거리(노트북 캠에서 1m)로 구분하고 각 5회씩 실시하였다.  
  결과는 <표1>, <표2>와 같다. 코드 수정 이후 근거리 테스트에서 다른 경우와 확연히 구분되는 평균 fps 향상이 있었다. 이는 얼굴인식 함수(face_classifier.detectMultiScale)의 minSize를 수정하여 연산량을 줄인 결과로 보인다.  
  테스트에서는 노트북 카메라의 성능을 고려하여 구간을 2개로 나누어서 수행하였으나 여러 개의 구간을 더욱 세분화한다면 추가적인 성능향상을 기대할 수 있을 것으로 보인다.  
  
    
<표1>  
![표1](https://user-images.githubusercontent.com/71097404/114285873-0632f080-9a8d-11eb-9e13-eb85b6bf53ec.JPG)  
<표2>  
![표2](https://user-images.githubusercontent.com/71097404/114285869-0206d300-9a8d-11eb-810a-bc2d5fa38d3d.JPG)
