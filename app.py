from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, g, Response
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from flask import jsonify
from MySQLdb.cursors import DictCursor
from functools import wraps
from PIL import Image
import numpy as np
import base64
import cv2
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = '169cc9856ed004364c82e430d8c62c16'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '123456'
app.config['MYSQL_DB'] = 'faceattend'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)


# Dataset generator 
def generate_dataset(nbr):
    
    """
train image by capturing 
User image from webcam 
"""
    face_classifier = cv2.CascadeClassifier("/resources/haarcascade_frontalface_default.xml")
     # FIXME  generating dataset in open cv
     # TODO 1. fixing a bug in generate_dataset function
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(img ,scaleFactor=1.1,minNeighbors=3)


        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        return cropped_face

    cap = cv2.VideoCapture(0)
    cur = mysql.connection.cursor()
    cur.execute("select ifnull(max(image_ID), 0) from image_dataset")
    row = cur.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Faceattend",face)
            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cur = mysql.connection.cursor()

            cur.execute("""INSERT INTO `image_dataset` (`image_id`, `image_person`) VALUES('{}', '{}')""".format(img_id, nbr))
            mysql.connection.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
            cap.release()
            cv2.destroyAllWindows()



# TODO 2
# capture = cv2.VideoCapture(0)
# mod = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# while True:
#     boolean,frame = capture.read()
#     if boolean ==True:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         coo = mod.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
        
#         for(x,y,w,h) in coo:
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
#         cv2.imshow("Faceattend",frame)
#         if cv2.waitKey(20) == ord('x'):
#             break
# capture.release()
# cv2.destroyAllWindows()   

# Open cv trainer  
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id1 = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id1)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/dashboard')


# generate frame by frame from camera
def face_recognition():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
            cur = mysql.connection.cursor()
            cur.execute("select b.prs_name "
                        "  from image_dataset a "
                        "  left join attendee b on a.image_NAME = b.attendee_ID "
                        " where image_ID = " + str(id))
            s = cur.fetchone()
            s = '' + ''.join(s)

            if confidence > 70:
                cv2.putText(img, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 500, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

# home page route
@app.route('/home')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data')
def data():
    # Create cursor
    cur = mysql.connection.cursor()

    # sql command to select attendee data
    cur.execute("select attendee_ID, attendee_NAME, attendee_FIELD, attendee_STATUS, attendee_TIME from attendee_data")
    data1 = cur.fetchall()

    return render_template('data.html', data=data1)


@app.route('/parent')
def parent():
    return render_template('video.html')


@app.route('/feature')
def feature():
    return render_template('feature.html')


@app.route('/about')
def about():
    return render_template('about.html')


# wtf form class
class RegisterForm(Form):
    name = StringField('Name', [validators.length(min=2, max=50), validators.DataRequired()])
    username = StringField('Username', [validators.length(min=4, max=25), validators.DataRequired()])
    email = StringField('Email ID', [validators.length(min=6, max=50), validators.DataRequired()])
    password = PasswordField('Password', [validators.DataRequired(),
                                          validators.EqualTo('confirm', message='Password do not match'),
                                          validators.length(min=8, max=30)])
    confirm = PasswordField('Confirm Password')


# register a new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        cur = mysql.connection.cursor()

        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)",
                    (name, email, username, password))

        mysql.connection.commit()

        

        flash('You are now registered successfully', 'success')

        return redirect(url_for('home'))
    return render_template('register.html', form=form)


# login route
@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        username = request.form['username']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user data by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        # if any row found 
        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords from the db
            if sha256_crypt.verify(password_candidate, password):
                # confirm password
                session['logged_in'] = True
                session['username'] = username

                flash('You are successfully logged in', 'success')
                return redirect(url_for('dashboard'))  # to be fixed
            else:
                error = 'Invalid password'
                return render_template('login.html', error=error)
           
        else:
            error = 'user not found'
            return render_template('login.html', error=error)

    return render_template('login.html')


# check if the user is logged in or not to restrict random routing
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))

    return wrap


# dashboard route
@app.route('/dashboard')
@is_logged_in
def dashboard():
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("select ifnull(max(attendee_ID) + 1, 101) from attendee_data")
    row = cur.fetchone()
    # nbr = row[0]
    # print(int(nbr))
    return render_template('dashboard.html')


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    cur = mysql.connection.cursor()
    cur.execute("""INSERT INTO `attendee_data` (`attendee_ID`, `attendee_NAME`, `attendee_FIELD`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mysql.connection.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('login'))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('datagenrator.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video')
def video():
    return render_template('video.html')


if __name__ == '__main__':
    app.run(debug=True)
