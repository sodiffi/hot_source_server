from flask import Response,jsonify,make_response
import json
from coder import MyEncoder 

def ret(result):    
    mes="成功" if result["success"] else "失敗"
    resultData=result["data"] if "data" in result else {}   
    return make_response(json.dumps({"D":resultData,"message":mes,"success":result["success"]}, cls=MyEncoder))
 