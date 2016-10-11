# -*- coding: UTF-8 -*-
import os

from flask import Flask, url_for, render_template, request, redirect, Response
from werkzeug import secure_filename
from infofile import InfoFile
from stream import VideoStream
import detection

app = Flask(__name__)
UPLOAD_PATH = '/var/www/flask/static/uploads'
ALLOWED_EXTENSIONS=set(['png','jpg','jpeg','gif','mp4'])
app.config['UPLOAD_PATH'] = UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

app.config['videoinfo'] = None
app.config['imginfo'] = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload/<info>', methods=['GET','POST'])
def upload(info):
    print info
    if request.method=='POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            return redirect(url_for('upload',info=info, filename=filename))
    app.config[info] = InfoFile(UPLOAD_PATH, request.args.get('filename'))
    print app.config[info]
    print app.config[info].getfilefullpath()
    print app.config[info].getfilepurename()
    print app.config[info].getfilefullname()
    print url_for('upload', info=info, filename=filename)
    return render_template('error.html')

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/test')
def test():
    return render_template('test.html')

if __name__ == '__main__':
    app.run()
