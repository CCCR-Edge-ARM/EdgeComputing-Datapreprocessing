import socket
import numpy
import cv2
import os
import datetime
import pymysql

# Edge Node IP Address & Port
UDP_IP = "192.168.0.4"
UDP_PORT = 9505
UDP_PORT2 = 9506

# UDP 프로토콜 소켓 통신
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock2.bind((UDP_IP, UDP_PORT2))

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
names = ['HSH', 'HS', 'HWJ', 'chs', 'ksw']

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

				# 얼굴 인식 될 시 온도 데이터 송신
				data2, addr = sock2.recvfrom(1024)
				# 수신한 온도데이터 디코딩
				thermal = data2.decode()
				# 온도데이터를 문자열에서 리스트로 변환
				thermal = thermal.split()
				# 문자로 변환되었던 픽셀의 온도 값들을 실수형으로 변환
				thermal = list(map(float, thermal))
				# 픽셀의 온도데이터 평균 (소수점 1자리까지 표현)
				thermal = sum(thermal, 0.0)/len(thermal)
				thermal = round(thermal,1)

				#현재시간
				time = datetime.datetime.now()

				# DB instance의 armtong db에 접속
				db = pymysql.connect(host='192.168.0.165', port=3306, user='stack', passwd='stack', db='armtong', charset='utf8')

				# SQL 실행 (TEMPERATURE 테이블 사용)
				cursor = db.cursor()
				sql = "select * from TEMPERATURE"
				cursor.execute(sql)
				# SQL 실행 (id, 측정온도, 측정시간, 측정장소를 입력)
				sql2 = "INSERT INTO TEMPERATURE (member_idx,temperature_tem,temperature_date,temperature_location,) VALUES (%s,%s,%s,%s);"
				cursor.execute(sql2,(id,thermal,time,"guro.seoul"))

				# Connection 객체의 commit() 메서드를 사용
				db.commit()
				# Connection 객체의 close() 메서드를 사용하여 DB 연결 종료
				db.close()

			else:

				# 저장되지 않은 사용자일 경우 unkwon으로 표시
				id = 999
				user = "unknown"
				confidence = "  {0}%".format(round(100 - confidence))

				# 얼굴 인식 될 시 온도 데이터 송신
				data2, addr = sock2.recvfrom(1024)
				# 수신한 온도데이터 디코딩
				thermal = data2.decode()
				# 온도데이터를 문자열에서 리스트로 변환
				thermal = thermal.split()
				# 문자로 변환되었던 픽셀의 온도 값들을 실수형으로 변환
				thermal = list(map(float, thermal))
				# 픽셀의 온도데이터 평균 (소수점 1자리까지 표현)
				thermal = sum(thermal, 0.0)/len(thermal)
				thermal = round(thermal,1)

				#현재시간
				time = datetime.datetime.now()
				
				# DB instance의 armtong db에 접속
				db = pymysql.connect(host='192.168.0.165', port=3306, user='stack', passwd='stack', db='armtong', charset='utf8')

				# SQL 실행 (TEMPERATURE 테이블 사용)
				cursor = db.cursor()
				sql = "select * from TEMPERATURE"
				cursor.execute(sql)
				# SQL 실행 (id, 측정온도, 측정시간, 측정장소를 입력)
				sql2 = "INSERT INTO TEMPERATURE (member_idx,temperature_tem,temperature_date,temperature_location,) VALUES (%s,%s,%s,%s);"
				cursor.execute(sql2,(id,thermal,time,"guro.seoul"))

				# Connection 객체의 commit() 메서드를 사용
				db.commit()
				# Connection 객체의 close() 메서드를 사용하여 DB 연결 종료
				db.close()

		k = cv2.waitKey(100) & 0xff # 'ESC'를 누르면 종료
		if k == 27:
			break
print("\n [INFO] Exiting Program")


