# Warehouse-System-with-YOLO
## 개요

- CCTV **영상**에서 특정 구역 내의 **객체**(사람, 지게차)를 **실시간**으로 **검지**하고, 위험 구역 진입과 같은 **이벤트**를 **자동**으로 **인지**하는 시스템
- **인지된 이벤트**는 **데이터베이스**에 **기록**
- 사용자는 **웹 인터페이스**를 통해 **구역**을 설정하고, 발생한 **이벤트**를 **기록** 및 **관리** 가능

## 사용 도구/기술

- **Python**: 개발언어
- **Flask**: 웹 기반 인터페이스
- **YOLOv5**: 객체 검지
- **OpenCV**: 비디오 처리
- **SQLite**: 데이터베이스
- **Flask-SocketIO**: 실시간 알림 (웹 소켓 통신)
- **HTML, CSS, Bootstrap**: 웹페이지 탬플릿
- **Colab**: 모델학습
- **Visual Studio Code**: 코드 작성
- **Windows, Linux**: 운영체제

## 데이터베이스 설계

- 발생한 **이벤트의 종류**(위험구역 진입 등), **Confidence**, **발생 시간**을 Event **테이블**에 **기록**
- 이벤트가 발생할 때마다 Event 테이블에 추가
- 웹 페이지의 **결과 조회 페이지**에서 최신순으로 이벤트 로그 확인 가능
- Flask-SocketIO를 통해 메인 페이지에 실시간으로 이벤트 표시

## 객체 검지 및 이벤트 처리

- **YOLOv5** 모델을 사용하여 CCTV 영상 또는 지정된 영상에서 **객체**(사람, 지게차)를 **실시간**으로 **검지**.
- **이벤트 인지 로직**:
  - **위험구역 진입**: 사람(작업자 등)이 설정된 구역에 들어오면 인지.
  - **제한구역 진입**: 지게차가 설정된 구역에 들어오면 인지.
  - **근접 경보**: 지게차와 사람의 거리가 일정 수준 미만일 경우 인지.

## 웹 애플리케이션 구조

- Flask를 사용하여 웹페이지와 기타 기능 제공. 주요 페이지는 다음과 같음:
  - **로그인 페이지**: 관리자 로그인 기능 제공. 로그인 상태가 아닌 경우, 다른 페이지 접근 시 로그인 페이지로 리다이렉트.
  - **메인 페이지**: 실시간 CCTV 영상 표시, 검지된 이벤트 실시간 표시, 구역 설정 가능.
  - **결과 조회 페이지**: 데이터베이스에 기록된 이벤트를 최신 순으로 조회 가능.
  - **설정 페이지**: 위험구역 및 제한구역 설정 가능.

## 주요 구현 흐름

![image](https://github.com/user-attachments/assets/d3ec8124-66b3-4fa3-ad5d-1519ae4e7dd0)
![image](https://github.com/user-attachments/assets/df3ba53f-7983-4e71-aa19-5ed4951728ba)

## 파일 목록

### detection.py - 객체 검지, 이벤트 인지 수행
### video_stream.py - 비디오 스트림 처리 수행 
### events.py - 이벤트 처리, 저장, 사용자 알림 수행
### database.py - 데이터베이스 파일 생성 & 연결, 테이블 클래스 정의 수행
### app.py - Flask 파일, 다양한 기능들을 통합하고 웹을 통해 동작하도록 함
### login.html - 로그인 페이지 템플릿
### index.html - 메인 페이지 템플릿
### events.html - 결과 조회 페이지 템플릿
### best.pt - YOLO 객체 검지 모델
### test.mp4 - 테스트 용 영상
### requirements.txt - 실행에 필요한 패키지 목록

## 실행 방법(리눅스에서만 가능)

1. 프로젝트 클론: `git clone https://github.com/junny1117/Warehouse-System-with-YOLO`
2. 필요한 패키지 설치: `pip install -r requirements.txt`
3. Flask 서버 실행: `flask run`
4. 브라우저에서 `127.0.0.1:5000`으로 접속

## 실행 결과 이미지
![스크린샷 2024-10-22 144901](https://github.com/user-attachments/assets/6379cc3f-c1a3-4097-8b78-8143b566ef88)
![스크린샷 2024-10-22 152600](https://github.com/user-attachments/assets/e223f8b0-d4cf-40a3-911b-7725a638c4dd)
![image](https://github.com/user-attachments/assets/8b7a40ed-267d-45c2-87c5-c7ef72a2487e)
![스크린샷 2024-10-22 142513](https://github.com/user-attachments/assets/8f7d2c93-0971-4330-829f-cc3a4f591586)

