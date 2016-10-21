# Flask

Graduation piece

---

ubuntu 14.04 on AWS EC2
python 2.x

## structure

flask
├── app.py
│   ├── stream.py
│   ├── detection.py
│   ├── facetracking.py
│   └── infofile.py
│
├── templates/
│   ├── js/
│   ├── layout.html
│   ├── upload.html
│   ├── image.html
│   └── video.html
│
├── static/
│   ├── uploads/
│   └── css
│
├── .gitignore
├── README.md
├── favicon.ico
└── uwsgi.ini


#### Prerequisite

[CPU Only](//lastone9182.github.io/2016/08/29/aws-flask.html)
[GPU](//lastone9182.github.io/2016/09/16/aws-caffe.html)

* Nginx
* Uwsgi
* OpenCV ( - GPU )
* TensorFlow ( - GPU )
* ( Caffe - GPU )
* FFMPEG

#### Recommend using screen

```bash
screen -S newscreen
```

> `/var/www/flask` is root directory

## Configuration

```bash
mkdir /var/www
chown -R ubuntu:ubuntu /var/www

cd /var/www
git clone https://github.com/lastone9182/flask
ln -s /var/www/flask ~/flask
```

> Nginx(or Apache) install and configuration Required !!

## C++ extension Install & Server Launch

```bash
# package install required
sudo apt-get install python-dev

# $flask : clone directory
cd $flask/Cextension

# if using virtualenv
python setup.py install

# no using virtualenv
sudo -H python setup.py install

# and... nginx start 
cd $flask
uwsgi --ini uwsgi.ini
```
