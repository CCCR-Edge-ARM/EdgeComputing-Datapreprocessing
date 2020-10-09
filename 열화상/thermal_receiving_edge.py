import socket

# Edge Node IP Address & Port
UDP_IP = "192.168.0.4"
UDP_PORT2 = 9506

# UDP 프로토콜 소켓 통신
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2.bind((UDP_IP, UDP_PORT2))


while(1):

	# 온도 데이터 수신
    data2, addr = sock2.recvfrom(1024)

    # 수신한 온도데이터 디코딩
    thermal=data2.decode()

    # 온도데이터를 문자열에서 리스트로 변환
    thermal=thermal.split()

    # 문자로 변환되었던 픽셀의 온도 값들을 실수형으로 변환
    thermal=list(map(float, thermal))

    # 픽셀의 온도데이터 평균 (소수점 1자리까지 표현)
    thermal=sum(thermal, 0.0)/len(thermal)
    thermal=round(thermal,1)

    # 온도값 출력
    print(thermal)