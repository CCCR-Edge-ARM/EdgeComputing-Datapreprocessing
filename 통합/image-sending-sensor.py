import socket
import os
import cv2

# Edge Node IP Address & Port
UDP_IP = '192.168.0.4'
UDP_PORT = 9505

# UDP 프로토콜 소켓 통신
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 얼굴인식 객체 및 학습파일 참조
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

#id 카운트 시작
id = 0

# id에 저장할 사용자 설정
names = ['0', 'HS', 'HWJ', 'HSH', 'FUNFUN']

# 카메라 모듈의 영상테이터 
cap = cv2.VideoCapture(0)
cap.set(3, 640) # 영상의 크기(넓이)
cap.set(4, 480) # 영상의 크기(높이)

while True:

    # 영상데이터를 frame에 저장
    ret, frame = cap.read()
    d = frame.flatten()
    s = d.tostring()
    
    # 얼굴 인식
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        # 출력 영상에 사용자 출력
        cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    # 영상 출력
    cv2.imshow('camera', frame)

    # 분할된 데이터 송신 (데이터의 크기 = 640(영상 너비)*480(영상 높이)*3(RGB)/20(분할 수) = 46080)
    for i in range(20):
        sock.sendto(bytes([i]) + s[i*46080:(i+1)*46080], (UDP_IP, UDP_PORT))

        k = cv2.waitKey(100) & 0xff # 'ESC'를 누르면 종료
        if k == 27:
            break
