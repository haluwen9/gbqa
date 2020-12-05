import os
import io

from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from flask import redirect
from flask import url_for
from flask import send_from_directory, send_file
from werkzeug.utils import secure_filename

from gbqa import GBQA

app = Flask(__name__)
gbqa = GBQA()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype = 'image/x-icon')

@app.route("/ask", methods = ["POST"])
def ask():
    question = request.json["question"].strip()

    return gbqa.ask(question)

if __name__ == "__main__":
    app.run()
