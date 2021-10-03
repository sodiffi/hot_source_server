from flask import Flask, request
from util import (ret)
from module import nasaModule
from heat_index import calculate
app = Flask(__name__)

app = Flask(__name__)

@app.route("/temp")
def hello_world():
    city = request.args.get("city")
    if city != "":
        temps = nasaModule.getTemp(city)
        if(temps != []):
            temp = temps["data"][0]["temp"]
            hum = temps["data"][0]["hum"]
            heat_index = calculate.from_celsius(temp-273.15, hum)

        return ret({"data": {"temps": temps["data"], "heat_index": heat_index, "suggestion": "suggestion", "heat_percent": 0}, "message": "seach success", "success": True})
    else:
        return ret({"message": "search fail", "success": False})
