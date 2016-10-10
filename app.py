# -*- coding: UTF-8 -*-
import os

# flask import
from flask import Flask, url_for, render_template, request, redirect, Response
from werkzeug import secure_filename

# filename class
from infofile import InfoFile

# opencv class
from stream import VideoStream

# opencv image
import detection

UPLOAD_PATH='/var/www/flask/static/uploads'
ALLOWED_EXTENSIONS=set(['txt','pdf','png','jpg','jpeg','gif','mp4'])
app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploaded', methods=['GET','POST'])
def uploaded():
        # class globalization
        global info

	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
                        
                        # file name into class
                        info = InfoFile(UPLOAD_PATH, file.filename)
			
                        file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			return redirect(url_for('uploaded', filename=filename))
        
        #info = InfoFile(UPLOAD_PATH, request.args.get('filename'))
	filefullpath = info.getfilefullpath()
        filename = info.getfilefullname()
        ext = info.getfileext()


	if ext in ('png','jpg','jpeg','gif'):
                detection.detectionImage(UPLOAD_PATH, filename)
                return redirect(url_for('image'))
		#return render_template('image.html')
	elif ext in ('mp4'):
                #return Response(gen(VideoStream(info.getfilefullpath())), mimetype='multipart/x-mixed-replace; boundary=frame')
	        #return render_template('video.html')
                return redirect(url_for('video', filename=filename))
	else:
		return render_template('error.html')

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/stream')
def stream():
    return Response(gen(VideoStream(info.getfilefullpath())), mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/test')
def test():
    return render_template('test.html')

# Video Streaming
def gen(stream):
    while True:
        frame = stream.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0')

