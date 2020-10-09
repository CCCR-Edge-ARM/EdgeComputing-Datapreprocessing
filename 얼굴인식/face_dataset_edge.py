import socket
import numpy
import cv2
import os
import datetime

# Edge Node IP Address & Port
UDP_IP = "192.168.0.4"
UDP_PORT = 9505

# # UDP 프로토콜 소켓 통신
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# 영상 분할을 저장하기 위한 리스트 생성
s = [b'\xff' * 46080 for x in range(20)]

# 얼굴인식 객체 참조
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# id 입력
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# 사용자의 얼굴 데이터를 수집 및 저장
count = 0
while True:
    picture = b''

    # 라즈베리파이에서 데이터 수신
    data, addr = sock.recvfrom(46081)
    s[data[0]] = data[1:46081]

    # 리스트로 들어온 분할된 데이터를 다시 합쳐서 이미지로 변환
    if data[0] == 19:
        for i in range(20):
            picture += s[i]
        frame = numpy.fromstring(picture, dtype=numpy.uint8)
        frame = frame.reshape(480, 640, 3)

        # 얼굴 인식 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # 이미지에서 얼굴이 인식되면 얼굴부분 추출
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            # 인식된 얼굴부분을 저장
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(datetime.datetime.now()) + ".jpg", gray[y:y+h,x:x+w])

        k = cv2.waitKey(100) & 0xff # 'ESC'를 누르면 종료
        if k == 27:
            break
        elif count >= 30: # 30장의 샘플 이미지 생성 후 종료
            break
print("\n [INFO] Exiting Program")
