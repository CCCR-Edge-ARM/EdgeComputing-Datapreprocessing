import socket
import numpy
import cv2
import os
import datetime

# Edge Node IP Address & Port
UDP_IP = "192.168.0.4"
UDP_PORT = 9505

# UDP 프로토콜 소켓 통신
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# 영상 분할을 저장하기 위한 리스트 생성
s = [b'\xff' * 46080 for x in range(20)]

# 얼굴인식 객체 및 학습파일 참조
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

#id 카운트 시작
id = 0

# id에 저장할 사용자 설정: example ==> loze: id=1,  etc
names = ['HSH', 'HS', 'HWJ']

while True:        
	picture = b''
	
	# 분할된 데이터 수신
	data, addr = sock.recvfrom(46081)
	s[data[0]] = data[1:46081]

	# 분할된 데이터를 다시 합쳐서 이미지로 변환
	if data[0] == 19:
		for i in range(20):
			picture += s[i]
		frame = numpy.fromstring(picture, dtype=numpy.uint8)
		frame = frame.reshape(480, 640, 3)

		# 얼굴 인식
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                   
		faces = faceCascade.detectMultiScale( 
			gray,
			scaleFactor = 1.2,
			minNeighbors = 5,
			#minSize = (int(minW), int(minH)),
			)
		for(x,y,w,h) in faces:
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

			# 사용자 인식
			id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
			if (confidence < 100):
				user = names[id]
				confidence = "  {0}%".format(round(100 - confidence))
				
				#이름, 날짜, 장소 출력
				print(user, datetime.datetime.now(), "guro.seoul")
			else:

				# 저장되지 않은 사용자일 경우 unkwon으로 표시
				user = "unknown"
				confidence = "  {0}%".format(round(100 - confidence))
				
				#이름, 날짜, 장소 출력
				print(user, datetime.datetime.now(), "guro.seoul")

		k = cv2.waitKey(100) & 0xff # 'ESC'를 누르면 종료
		if k == 27:
			break
print("\n [INFO] Exiting Program")


