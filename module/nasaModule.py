from module.db import DB

import json


def getTemp(city):
    sqlstr = f"select time,hum,temp from record where cityName='{city}' order by time desc limit 7"
    return DB.execution(DB.select, sqlstr)
