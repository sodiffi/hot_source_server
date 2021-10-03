from module.db import DB

import json


def home():
    sqlstr = "select * from "
    return DB.execution(DB.select, sqlstr)
