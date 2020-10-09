import socket
import cv2
import os
from Adafruit_AMG88xx import Adafruit_AMG88xx

# Edge Node IP Address & Port
UDP_IP = '192.168.0.4'
UDP_PORT = 9506
UDP_PORT2 = 9506

# UDP 프로토콜 소켓 통신
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock2.bind((UDP_IP, UDP_PORT2))

# 카메라 모듈의 영상테이터 
cap = cv2.VideoCapture(0)
cap.set(3, 640) # 영상의 크기(넓이)
cap.set(4, 480) # 영상의 크기(높이)

while True:

	# 영상데이터를 frame에 저장
    ret, frame = cap.read()

    # 상하 반전
    frame = cv2.flip(frame, -1)

    d = frame.flatten()
    s = d.tostring()

    # 영상 출력
    cv2.imshow('camera', frame)

    # 온도 데이터 저장(픽셀의 온도값들이 리스트로 저장)
    list_thermal = sensor.readPixels()

    # 픽셀의 온도값을 실수형에서 문자로 변환
    for i in range(len(list_thermal)):
        list_thermal[i]=str(list_thermal[i])

    # 온도 데이터를 리스트에서 문자열로 변환
    thermal = (' '.join(list_thermal))

    # 온도 테이터 송신(문자열을 인코딩해야 송신가능)
    sock2.sendto(thermal.encode(),(UDP_IP,UDP_PORT))

    # 분할된 데이터 송신
    for i in range(20):
        sock.sendto(bytes([i]) + s[i*46080:(i+1)*46080], (UDP_IP, UDP_PORT))
        
        k = cv2.waitKey(100) & 0xff # 'ESC'를 누르면 종료
        if k == 27:
            break