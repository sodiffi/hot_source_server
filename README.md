# HotSource_server

app client link: https://github.com/sodiffi/hot_Source_app

CONTENTS OF THIS FILE
---------------------

- [Background](#Background)
- [Install](#Install)


## Background
The Python Flask framework was used to construct a web server. When the city data is received via mobile phone and added to our database, the past seven days temperature and the calculated heat index value will be returned to the mobile phone. 
>We used Python Torch(Pytorch) to set up a machine learning model. It can help us predict daily future weather. Using the data, the future probability of heat waves can be predicted in advance.

The data preprocessing comes from the result of cross analysis of two data found on the Internet:
The city position coordinates
(https://gist.github.com/dannymorris/d28665a8b5e58f7eb6d8e065e04b1231#file-worldcities-csv). 
The historical monthly average temperature (https://www.kaggle.com/sudalairajkumar/daily-temperature-of-major-cities)
By using these data, we can filter out the highest historical temperature of the city along with its position coordinates.

These data will be used to further estimate the probability of heat waves.

In designing this machine learning model, we adopted the Linear Regression Model in Pytorch, chose the Mean Squared Error (MSE) as a loss criterion, and used Stochastic Gradient Descent(SGD) as the best formula.

By using data provided by NASA (https://disc.gsfc.nasa.gov/datasets/AIRS3STD_7.0/summary), 
we can predict future temperature and humidity, and move on to calculate the heat index value.

## Install
Click the green Clone, then press Download. The selected files will be downloaded in ZIP files.
###  Requirements
 * [Flask](https://pypi.org/project/Flask/)
 * [mysql-connector](https://pypi.org/project/mysql-connector/)
 * [PyTorch](https://pypi.org/project/torch/)
 * [numpy](https://pypi.org/project/numpy/)
 * [netCDF4](https://pypi.org/project/netCDF4/)
 * [matplotlib](https://pypi.org/project/matplotlib/)