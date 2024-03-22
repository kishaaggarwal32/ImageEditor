from flask import Flask
import cv2 as cv
import os
import numpy as np
from flask_cors import CORS, cross_origin
from flask import make_response, render_template, jsonify, request, redirect
import requests
import uuid
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RESULT_FOLDER = './results'
app.config['RESULT_FOLDER'] = RESULT_FOLDER

novu_url = "https://api.novu.co"
novu_api_key = os.environ.get('NOVU_API_KEY')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


def detect_face(img):
    face_cascade = cv.CascadeClassifier('front_face.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face = faces[0]
    x, y, w, h = face
    face_img = img[y:y + h, x:x + w]
    # return cropped face image
    return face_img


def readfile(file_path):
    with open(file_path, 'rb') as file:

        encode_string = base64.b64encode(file.read())
        encode_string = encode_string.decode('utf-8')
        return encode_string


def send_email(email, file_path):

    print(f"Email: Sending email ")
    headers = {
        'Authorization': f'ApiKey {novu_api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    data = {
        "name": "emailerworkflow",
        "to": {
            "subscriberId": str(uuid.uuid4()),
            "email": email
        },
        "payload": {
            "results": "Processed Image",
            "attachments": [
                {
                    "file": readfile(file_path),
                    "name": "processed_image.jpg",
                    "mime": "image/jpeg"

                }
            ]
        }
    }
    print(data)
    try:
        response = requests.post(novu_url + '/v1/events/trigger', headers=headers, json=data)
    except requests.exceptions.RequestException as e:
        print(e)
        return "Failed to send email"
    except Exception as e:
        print(e)
        return "Failed to send email"
    return "Email sent successfully"


@app.route('/')
@cross_origin()
def hello_world():
    return 'Hello, World!'


@app.route('/home')
@cross_origin()
def home():  # put application's code here
    return render_template('index.html')


@app.route('/face_editor', methods=['POST'])
@cross_origin()
def face_editor():
    # get Image file
    if 'file' not in request.files:
        return 'No file found'
    file = request.files['file']
    # read the image
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv.imread(filepath)
    print(f"Image shape: {img.shape}")
    data = request.form
    command = data['command']
    print(f"Command: {command}")
    email = data['emailID']

    # face detection
    face_image = detect_face(img)
    response = None

    if command == 'blur':
        # apply blur effect
        blurred_face = cv.GaussianBlur(face_image, (99, 99), 30)
        cv.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'blurred_face_' + filename), blurred_face)
        response = jsonify(
            {'processed_image_url': ('./results/' + 'blurred_face_' + filename)})

    elif command == 'grayscale':
        # apply grayscale effect
        gray_face = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
        cv.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'gray_face_' + filename), gray_face)
        response = jsonify(
            {'processed_image_url': ('./results/' + 'gray_face_' + filename)})

    elif command == 'sharpen':
        # apply to sharpen effect
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_face = cv.filter2D(face_image, -1, kernel)
        cv.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'sharpened_face_' + filename), sharpened_face)
        response = jsonify(
            {'processed_image_url': ('./results/' + 'sharpened_face_' + filename)})

    elif command == 'edge':
        # apply edge effect
        edges = cv.Canny(face_image, 100, 200)
        cv.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'edge_face_' + filename), edges)
        response = jsonify({'processed_image_url': ('./results/' + 'edge_face_' + filename)})

    elif command == 'cartoon':
        # apply cartoon effect
        gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)
        color = cv.bilateralFilter(face_image, 9, 300, 300)
        cartoon = cv.bitwise_and(color, color, mask=edges)
        cv.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'cartoon_face_' + filename), cartoon)
        response = jsonify(
            {'processed_image_url': ('./results/' + 'cartoon_face_' + filename)})

    elif command == 'rotate':
        # rotate the image
        angle = 90
        (h, w) = face_image.shape[:2]
        center = (w / 2, h / 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_face = cv.warpAffine(face_image, M, (w, h))
        cv.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'rotated_face_' + filename), rotated_face)
        response = jsonify(
            {'processed_image_url': ('./results/' + 'rotated_face_' + filename)})

    response_email = send_email(email, response.json['processed_image_url'])
    return 'Image processed successfully.'


if __name__ == '__main__':
    app.run()
