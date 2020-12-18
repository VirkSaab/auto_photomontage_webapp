import os, io, cv2
import numpy as np
from flask import Flask, render_template, request, redirect, make_response, jsonify



def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    images = {}# Store added images

    # a simple page that says hello
    @app.route('/', methods=["POST", "GET"])
    def homepage():
        # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask        
        if request.files:
            #TODO: Securing file uploads, go to section with same name in https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
            for image in request.files.getlist("file"):
                if image.filename != "":
                    in_memory_file = io.BytesIO()
                    image.save(in_memory_file)
                    img = np.fromstring(in_memory_file.getvalue(), dtype='uint8')
                    img = cv2.imdecode(img, 1)
                    images[image.filename] = img
                    print(f"homepage function: {image.filename}:", img.shape)
            return redirect(request.url)
        return render_template("home.html")


    @app.route("/output", methods=["POST"])
    def output():
        req = request.get_json()

        if req["runButton"] == "clicked":
            #TODO: Add processing function here
            print(len(images))
            # return the output image
            res = make_response("output image", 200)
        else:
            res = make_response("Error", 201)
        return res

    return app


"""
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run
"""
