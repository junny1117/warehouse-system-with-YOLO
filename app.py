import sys
import os
import cv2
import torch
import datetime
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
from pathlib import Path
from python.video_stream import VideoStream
from python.detection import ObjectDetector
from python.database import init_db, db_session, Event
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app)

# Dummy credentials for login
USERNAME = 'admin'
PASSWORD = '1234'

# Video stream
video_stream = VideoStream("test.mp4")

weights_path = 'best.pt'
roi_dangerzone = (100, 200, 300, 400)
roi_collision_caution = (400, 500, 300, 400)
dangerzone_time_threshold = 5
parking_time_threshold = 10  
detector = ObjectDetector(weights_path, roi_dangerzone, roi_collision_caution, dangerzone_time_threshold, parking_time_threshold)

tracked_objects = {}
tracked_events = {}

@app.before_request
def setup():
    init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid Credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    global roi_dangerzone, roi_collision_caution
    if request.method == 'POST':
        roi_dangerzone = (
            int(request.form['dangerzone_x']),
            int(request.form['dangerzone_y']),
            int(request.form['dangerzone_width']),
            int(request.form['dangerzone_height'])
        )
        roi_collision_caution = (
            int(request.form['collision_caution_x']),
            int(request.form['collision_caution_y']),
            int(request.form['collision_caution_width']),
            int(request.form['collision_caution_height'])
        )
        detector.roi_dangerzone = roi_dangerzone  # ObjectDetector 인스턴스의 ROI 업데이트
        detector.roi_collision_caution = roi_collision_caution  # ObjectDetector 인스턴스의 ROI 업데이트
        return redirect(url_for('index'))
    return render_template('settings.html')

@app.route('/select_roi')
@login_required
def select_roi():
    global roi_dangerzone
    frame = video_stream.get_frame()

    scale_percent = 50  # 프레임 크기 조정 (예: 50%로 축소)
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    roi_dangerzone_resized = cv2.selectROI("Select dangerzone ROI", resized_frame, fromCenter=False, showCrosshair=True)
    roi_dangerzone = (
        int(roi_dangerzone_resized[0] / scale_percent * 100),
        int(roi_dangerzone_resized[1] / scale_percent * 100),
        int(roi_dangerzone_resized[2] / scale_percent * 100),
        int(roi_dangerzone_resized[3] / scale_percent * 100)
    )
    cv2.destroyWindow("Select dangerzone ROI")

    detector.roi_dangerzone = roi_dangerzone  # ObjectDetector 인스턴스의 ROI 업데이트

    return redirect(url_for('index'))


@app.route('/select_roi2')
@login_required
def select_roi2():
    global roi_collision_caution
    frame = video_stream.get_frame()

    scale_percent = 50  # 프레임 크기 조정 (예: 50%로 축소)
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    roi_collision_caution_resized = cv2.selectROI("Select No Parking ROI", resized_frame, fromCenter=False, showCrosshair=True)
    roi_collision_caution = (
        int(roi_collision_caution_resized[0] / scale_percent * 100),
        int(roi_collision_caution_resized[1] / scale_percent * 100),
        int(roi_collision_caution_resized[2] / scale_percent * 100),
        int(roi_collision_caution_resized[3] / scale_percent * 100)
    )
    cv2.destroyWindow("Select No Parking ROI")

    detector.roi_collision_caution = roi_collision_caution  # ObjectDetector 인스턴스의 ROI 업데이트

    return redirect(url_for('index'))


@app.route('/events')
@login_required
def events():
    events = db_session.query(Event).order_by(Event.timestamp.desc()).all()  # 최신 이벤트부터 정렬
    return render_template('events.html', events=events)

def gen_frames():
    global tracked_objects, tracked_events
    while True:
        frame = video_stream.get_frame()
        if frame is None:
            break

        frame, detected_events = detector.detect_and_draw(frame)

        for event in detected_events:
            event_id = event['type'] + str(int(event['신뢰도'] * 100))  # 이벤트를 구분하는 ID 생성

            # 새로운 이벤트가 트래킹 중이 아니면 처리
            if event_id not in tracked_events:
                print(f"New event detected: {event_id}")  # 디버깅용 로그
                new_event = Event(
                    label=event['type'],
                    confidence=event['신뢰도'],
                    timestamp=datetime.datetime.now()
                )
                db_session.add(new_event)
                db_session.commit()
                socketio.emit('new_event', {
                    'label': new_event.label,
                    'confidence': new_event.confidence,
                    'timestamp': new_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
                tracked_events[event_id] = new_event.timestamp  # 이벤트를 트래킹에 추가

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    socketio.run(app, debug=True)
