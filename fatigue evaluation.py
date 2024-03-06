#=========================  READ_ME  ===========================#

####### 1. 아나콘다로 python 3.11 버전으로 가상환경 생성 후에 실행해주세요(파이썬 3.11 다운 필요)
#######
#######    conda create -n {사용할 이름} python=3.11
#######    pip install scipy imutils dlib
####### 
####### 
####### 2.data 아래 normal, sleep_eye, sleep_mouth, weights 4개의 폴더가 있고, 
#######   weights를 제외한 각 폴더 안에 original 폴더를 만들어 주세요
#######
#######    폴더 트리 구조
#######    root --- data --- normal --- original
#######                  |
#######         		 |-- sleep_eye --- original
#######        		 	 |
#######        			 |-- sleep_mouth --- original
#######        			 |
#######                  --- weights
#######
#######
####### 3. 이미지 저장경로를 본인이 저장할 곳에 맞게 바꿔주세요
#######    DATA_URI의 경로와 CURRENT_PATH의 할당할 변수를 바꿔주시면 됩니다.
#######
#######
####### 4. 캠은 지급된 웹캠으로 사용하면 화질이 안좋아서 인식이 잘 안됩니다. 
#######    웬만하면 사제 캠 사용해주셔야 합니다.
#######
#######
####### 5. 가이드라인 선을 본인 얼굴에 맞게 조정해주세요.
#######    (왼쪽 위 x좌표, 왼쪽 위 y좌표, width, height) 순으로, 
#######    대부분은 width와 height만 조절해주시면 되겠습니다.
#######
#######
####### 6. 캠이 켜졌으면 스페이스바를 누르면 저장됩니다.
#######    해당 폴더에 라벨링 후 배경까지 날린 사진, 
#######    해당 폴더 아래 original 폴더에 라벨링만 된 사진이 저장됩니다.

#=========================  READ_ME  ===========================#


# importing the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os

import torch
import torch.nn.functional as F


#calculating eye aspect ratio
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the vertical
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear

#calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
	# compute the euclidean distances between the horizontal
	X   = dist.euclidean(mou[0], mou[6])
	# compute the euclidean distances between the vertical
	Y1  = dist.euclidean(mou[2], mou[10])
	Y2  = dist.euclidean(mou[4], mou[8])
	# taking average
	Y   = (Y1+Y2)/2.0
	# compute mouth aspect ratio
	mar = Y/X
	return mar

def count_files_in_folder(folder_path):
    # 해당 폴더 내 모든 파일과 디렉토리 목록을 가져옵니다.
    all_files = os.listdir(folder_path)
    
    # 파일 개수를 저장할 변수를 초기화합니다.
    file_count = 0
    
    # 각 항목에 대해 파일 여부를 확인하고 파일인 경우 개수를 증가시킵니다.
    for item in all_files:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_count += 1
    
    return file_count

# 검출된 부분만 남기고 전부 흰색으로 변경
def labeling(frame):
    # RGB(255,0,0) 색상 범위 정의 (OpenCV는 BGR 순서를 사용하므로 (0, 0, 255))
	lower_green = np.array([0, 255, 0])
	upper_green = np.array([0, 255, 0])
	lower_red = np.array([255, 0, 0])
	upper_red = np.array([255, 0, 0])
 
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# 정의한 색상 범위에 따라 마스크 생성
	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask1 = cv2.inRange(hsv, lower_red, upper_red)

	mask = cv2.bitwise_or(mask, mask1)

	# 마스크를 사용하여 원본 이미지에서 색상을 추출
	res = cv2.bitwise_and(frame, frame, mask=mask)

	# 추출된 색상을 제외한 나머지 부분을 흰색으로 만듦
	res[np.where((res == [0,0,0]).all(axis=2))] = [255, 255, 255]
 
	return res

######################
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()    # CNN의 부모 클래스(torch.nn.Module) 호출
        
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(640 // 4 * 480 // 4 * 64, 3, bias=True)

        # 전결합층 한정으로 가중치 초기화 
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
        ### 세이비어 초기화: 가중치 초기화가 모델에 영향을 미침에 따라 초기화 방법 제안.
        ### 방법은 2가지 균등분포 or 정규분포로 초기화. 이전층의 뉴런 개수 / 다음층의 뉴런개수 사용하여 균등분포 범위 정해서 초기화
        ### He 초기화 방법도 있음. 
        ### sigmoid or tanh  사용할 경우 세이비어 초기화 효율적
        ### ReLU 계열 함수 사용할 때는 He 초기화 방법 효율적
        ### ReLU + He 초기화 방법이 좀 더 보편적임

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        self.out = out
        return out
     
     
# 이미지 저장 경로
DATA_URI = './data/'   ##### 본인 환경에 맞게 변경
NORMAL_FACE_PATH = DATA_URI + 'normal'
SLEEP_EYE_PATH = DATA_URI + 'sleep_eye'
SLEEP_MOUTH_PATH = DATA_URI + 'sleep_mouth'
TEST_PATH = DATA_URI + 'test'

CURRENT_PATH = NORMAL_FACE_PATH   ##### 사진 종류 변경 시 변수만 변경

COUNTER = count_files_in_folder(CURRENT_PATH)  

# 미리 학습된 모델 파일 경로
MODEL_PATH = './data/weights/30-7/ver1/model_state_dict.pt'

# 모델 불러오기
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 이미지 분류 함수 정의
def classify_image(image):
	# 이미지를 텐서로 변환
	image_tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
	# 모델에 입력하여 예측 수행
	with torch.no_grad():
		outputs = model(image_tensor)
		print(outputs, end=" ")
  
    # 예측 결과 중 확률이 가장 높은 클래스 선택
	_, predicted = torch.max(outputs, 1)


	return predicted.item()  # 클래스 인덱스 반환

######################






camera = cv2.VideoCapture(0)
predictor_path = './vsc/ai/shape_predictor_68_face_landmarks.dat'

width=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# grab the indexes of the facial landmarks for the left and right eye
# also for the mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# loop over captuing video
while True: 
	# grab the frame from the camera, resize
	# it, and convert it to grayscale
	# channels)
	ret, frame = camera.read()
	frame = imutils.resize(frame, width=640)
	frame = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# 가이드라인(얼굴 크기에 따라 수치 조정)
	cv2.rectangle(frame, (width//3 + 50, height//6 + 120, 25, 20), (128, 128, 254), 2)
	cv2.rectangle(frame, (width//3 + 95, height//6 + 120, 25, 20), (128, 128, 254), 2)
	cv2.rectangle(frame, (width//3 + 65, height//6 + 175, 45, 20), (128, 128, 254), 2)
 
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouEAR = mouth_aspect_ratio(mouth)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0,255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)



	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	result = labeling(frame)
	cv2.rectangle(result, (width//3 + 50, height//6 + 120, 25, 20), (128, 128, 254), 2)
	cv2.rectangle(result, (width//3 + 95, height//6 + 120, 25, 20), (128, 128, 254), 2)
	cv2.rectangle(result, (width//3 + 65, height//6 + 175, 45, 20), (128, 128, 254), 2)
 
	for rect in rects:
		# 얼굴 랜드마크 그리기
		landmarks = predictor(gray, rect)
		# 랜드마크 좌표를 저장할 리스트 초기화
		landmark_points = []
		for n in range(0, 27):  # 랜드마크의 총 갯수
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			landmark_points.append((x, y))
			# 각 랜드마크를 초록색 점으로 그림
			cv2.circle(result, (x, y), 1, (0, 0, 0), -1)

		# 얼굴 랜드마크 간의 선을 그림
		for i in range(1, len(landmark_points)):
			if i == 17:
				cv2.line(result, landmark_points[0], landmark_points[i], (0, 255, 0), 1)
			elif i == 26:
				cv2.line(result, landmark_points[16], landmark_points[i], (0, 255, 0), 1)
				cv2.line(result, landmark_points[25], landmark_points[i], (0, 255, 0), 1)

			else:
				cv2.line(result, landmark_points[i - 1], landmark_points[i], (0, 255, 0), 1)
 
 
	cv2.imshow("labeled", result)

	# 이미지를 CNN 모델에 전달하여 분류 수행
	classification_result = classify_image(result)

	# 분류 결과에 따라 적절한 후속 작업 수행
	print(classification_result)


	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

  

# do a bit of cleanup
cv2.destroyAllWindows()
camera.release()