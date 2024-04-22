from .InputStreamGenerator import InputStreamGenerator

from flask import Flask

from transcriber import transcriber_app

def create_app():
    app = Flask(__name__)

    app.register_blueprint(transcriber_app.bp)
    return app