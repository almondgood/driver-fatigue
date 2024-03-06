import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('KakaoTalk_20240215_103140806_1.png')

# RGB(255,0,0) 색상 범위 정의 (OpenCV는 BGR 순서를 사용하므로 (0, 0, 255))
lower_green = np.array([0, 255, 0])
upper_green = np.array([0, 255, 0])
lower_red = np.array([255, 0, 0])
upper_red = np.array([255, 0, 0])

# 이미지를 HSV 색 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 정의한 색상 범위에 따라 마스크 생성
mask = cv2.inRange(hsv, lower_green, upper_green)
mask1 = cv2.inRange(hsv, lower_red, upper_red)

mask = cv2.bitwise_or(mask, mask1)

# 마스크를 사용하여 원본 이미지에서 색상을 추출
res = cv2.bitwise_and(image, image, mask=mask)

# 추출된 색상을 제외한 나머지 부분을 흰색으로 만듦
res[np.where((res == [0,0,0]).all(axis=2))] = [255, 255, 255]


# 결과를 보여줌
cv2.imshow('Result', res)
cv2.imwrite('result.jpg', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
