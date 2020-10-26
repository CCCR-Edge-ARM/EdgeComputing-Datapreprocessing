import socket
from Adafruit_AMG88xx import Adafruit_AMG88xx

# Edge Node IP Address & Port
UDP_IP = '192.168.0.4'
UDP_PORT = 9506

# UDP 프로토콜 소켓 통신
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 열화상 카메라의 온도 데이터
sensor = Adafruit_AMG88xx()

while(1):

	# 온도 데이터 저장(픽셀의 온도값들이 리스트로 저장)
    list_thermal = sensor.readPixels()

    # 픽셀의 온도값을 실수형에서 문자로 변환
    for i in range(len(list_thermal)):
        list_thermal[i]=str(list_thermal[i])

    # 온도 데이터를 리스트에서 문자열로 변환
    thermal = (' '.join(list_thermal))

    # 온도 테이터 송신(문자열을 인코딩해야 송신가능)
    sock2.sendto(thermal.encode(),(UDP_IP,UDP_PORT))