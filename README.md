# 운전자 피로 감지 시스템

# 개요

###  주제 : CNN을 활용한 운전자의 피로도 측정 및 알림
-   얼굴인식과 전처리 과정을 거친 이미지를 데이터로 가공하여 **CNN(Convolution Neural Network)** 모델로 특징을 추출한다.
-    CNN으로 학습된 모델을 실시간 영상에 적용하여 운전자의 상태를 분석해 졸음운전으로 인한 사고를 예방한다.

![image](https://github.com/almondgood/driver-fatigue/assets/88735581/65fed501-d067-43cc-b8e7-0ae716458739)

<br>

---

<br>

### 팀원
- 김용기
- 신혜원
- 이재경
- 정우택

<br>

---

<br>

### 일정 및 개발 계획

![image](https://github.com/almondgood/driver-fatigue/assets/88735581/353ec950-8c79-402a-817f-3c832697eb1c)

<br><br>

---

<br><br>



# 개발 과정

### 개발 환경

![image](https://github.com/almondgood/driver-fatigue/assets/88735581/e0261b0e-8fb4-4199-bee6-e6b1dd7f75d9)

<br>

---

<br>

### CNN 모델 구조

![image](https://github.com/almondgood/driver-fatigue/assets/88735581/865f9241-24ad-4b48-89a5-7dfc23f03567)

||입력 데이터|제1합성층|제2합성층|평탄화|
|:---:|:---:|:---:|:---:|:---:|
|가로|640|320|160|1228800|
|세로|480|240|120|1|
|높이|3|32|64|1|



<br>

---

<br>

### 모델 평가 절차

![image](https://github.com/almondgood/driver-fatigue/assets/88735581/bc879d28-6fd5-469e-87ec-33043330359f)

|정상 상태|
|:---:|
|![image](https://github.com/almondgood/driver-fatigue/assets/88735581/ceae1653-eede-4132-abfd-c17fef5fe86c)|
|정면을 제대로 주시 중인 상태|

|하품 중인 상태|
|:---:|
|![image](https://github.com/almondgood/driver-fatigue/assets/88735581/ef50e821-4499-485d-8595-7637098ca200)|
|입을 일정 크기 이상 벌린 상태|

|눈을 감은 상태|
|:---:|
|![image](https://github.com/almondgood/driver-fatigue/assets/88735581/a5a33908-013d-4497-86f5-777f06188122)|
|눈을 1초 이상 감고있는 상태|
