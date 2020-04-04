from flask import Flask
import base64

def encode_image(img_name):
    with open('fig_'+img_name+'.png', "rb") as image_file:
        return base64.b64encode(image_file.read())

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Siema radziu<h1>'

@app.route('/<img_name>')
def files(img_name):
    return encode_image(img_name)

if __name__ == '__main__':
    app.run()





