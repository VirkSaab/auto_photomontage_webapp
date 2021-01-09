import os, io, cv2
import numpy as np
from flask import Flask, render_template, request, make_response
from .main import *

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    if app.config["ENV"] == "production":
        app.config.from_object("config.ProductionConfig")
    elif app.config["ENV"] == "testing":
        app.config.from_object("config.TestingConfig")
    else:
        app.config.from_object("config.DevelopmentConfig")
    # app.config.from_mapping(
    #     SECRET_KEY='dev',
    #     DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    # )

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

    global images
    images = {}
    OUTPUT_DIR = "flaskr/static/imgs"
    OUTPUT_PATH = f"{OUTPUT_DIR}/processed.png"

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
        return render_template("home.html")


    @app.route("/output", methods=["POST"])
    def process_output():
        req = request.get_json()

        if req["runButton"] == "clicked":
            num_images = len(images)
            processing_complete = False
            #TODO: Add processing function here
            if num_images >= 2:
                output_image = make_collage(images)
                print(f"OUTPUT IMAGE: {output_image.shape}")
                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
                cv2.imwrite(OUTPUT_PATH, output_image)
                processing_complete = True
            # return the output image
            res_output = {
                # "output": output_image,
                "num_images": num_images,
                "processing_complete": processing_complete
            }
            res = make_response(res_output, 200)

        else:
            res = make_response("Error", 201)
        return res

    @app.route("/checks", methods=["POST"])
    def check_before_running():
        req = request.get_json()
        if req["remove"] == "remove":
            while len(images) != 0:
                try:
                    for k in images.keys():
                        del images[k]
                except RuntimeError:
                    pass
            if os.path.exists(OUTPUT_PATH):
                os.remove(OUTPUT_PATH)
        return make_response({"status": 200}, 200)

    return app


"""
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run
"""
