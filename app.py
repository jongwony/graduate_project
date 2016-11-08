# -*- coding: UTF-8 -*-
import os

# flask import
from flask import Flask, url_for, render_template, request, redirect, Response, make_response
from werkzeug import secure_filename

# filename class
from infofile import InfoFile

# opencv class
from stream import VideoStream

# opencv image
from detection import detectionImage, traceImage

import numpy as np


UPLOAD_PATH='/var/www/flask/static/uploads'
ALLOWED_EXTENSIONS=set(['png','jpg','jpeg','gif','mp4'])

# Flask
app = Flask(__name__)

# global variable config
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
app.config['UPLOAD_PATH'] = UPLOAD_PATH

# global variable 
app.config['fileinfo'] = None
app.config['tfinfo'] = None

# split extension
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Video Streaming
def gen(stream):
    while True:    
        frame = stream.get_frame()
        if not frame:
            break
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# upload module
@app.route('/upload/<info>', methods=['GET','POST'])
def upload(info):
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
            return redirect(url_for('upload', info=info, filename=filename))
    
    #### GET method ####
    # filename into global class
    app.config[info] = InfoFile(app.config['UPLOAD_PATH'], request.args.get('filename'))
    filename = app.config[info].getfilefullname()
    ext = app.config[info].getfileext()

    # redirect from extension
    if ext in ['png','jpg','jpeg','gif']:
        return redirect(url_for('image', info=info, filename=filename))
    elif ext in ['mp4']:
        return redirect(url_for('video', info=info, filename=filename))
    else:
        return redirect(url_for('error'))

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/image/<info>')
def image(info):
    if info == 'fileinfo':
        filename = app.config[info].getfilefullname()
        detectionImage(app.config['UPLOAD_PATH'], filename)
        return render_template('image.html')
    elif info == 'tfinfo':
        filename = app.config[info].getfilefullpath()
        roi_gray = traceImage(filename)
        VideoStream.queryimg = roi_gray
        return render_template('trackimg.html')
    else:
        return render_template('error.html')

@app.route('/video/<info>')
def video(info):
    return render_template('video.html')

@app.route('/stream')
def stream():
    filename = app.config['fileinfo'].getfilefullpath()
    return Response(gen(VideoStream(filename)), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0')

