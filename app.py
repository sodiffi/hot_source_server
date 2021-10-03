from flask import Flask, request
from util import (ret)
from module import nasaModule
app = Flask(__name__)


@app.route("/temp")
def hello_world():
    city = request.args.get("city")
    if city != "":
        return ret(nasaModule.getTemp(city))
    else:
        return ret({"message": "search fail", "success": False})
