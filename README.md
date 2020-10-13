# EdgeComputing-Datapreprocessing
- EdgeComputing의 데이터 프로세싱 개발 및 운영 스터디를 위한 저장소입니다.
- 가세한 프로젝트 프로젝트 진행 사항은 [Wiki](https://github.com/CCCR-Edge-ARM/EdgeComputing-Datapreprocessing/wiki)에 작성됩니다.



## Senser-Node - Raspberry Pi
### 프로젝트 내용
- 얼굴 인식<br>
      - Raspberry Pi Pi 카메라 모듈로 사람의 얼굴을 인식<br>
      - 사람의 얼굴이 인식될 때 사람의 이름, 찍힌 시간 등의 데이터를 추출 후 저장
- 열 측정<br>
      - 열 화상센서를 통해 사람의 열을 측정<br>
      - 카메라 모듈과 연동하여 인식된 사람의 열 데이터도 같이 저장
      
<br>      

## EdgeNode-Vraptor
### 프로젝트 내용
- 데이터 전처리<br>
   - SensorNode-Raspberry Pi에서 보내는 센서데이터(영상 데이터, 열 측정 데이터)를 수집
   - 들어온 센서 데이터를 전처리 하여 필요한 데이터만 Server에 전송
