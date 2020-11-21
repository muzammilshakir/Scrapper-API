import json
import uuid
import base64
import os
from flask import Flask
from flask import request
from model import predict_sound_emotion

app = Flask(__name__)


@app.route('/audio/analyze', methods=['POST'])
def analyze_audio():
    audio_base_64 = request.json['audioBase64']
    filename, file_extension = os.path.splitext(request.json['fileName'])

    if file_extension != '.wav':
        data = {'message': 'Please upload wav file'}
        response = app.response_class(response=json.dumps(data), status=400, mimetype='application/json')
        return response

    if audio_base_64 == "":
        data = {'message': 'Base64 cannot be empty'}
        response = app.response_class(response=json.dumps(data), status=400, mimetype='application/json')
        return response

    filepath = os.path.abspath(os.path.join('temporary', str(uuid.uuid4()) + file_extension))
    decode_string = base64.b64decode(audio_base_64)

    with open(filepath, 'wb') as audio_file:
        audio_file.write(decode_string)

    results = predict_sound_emotion(filepath)
    os.remove(filepath)
    return results


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
