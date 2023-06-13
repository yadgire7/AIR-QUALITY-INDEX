import sys
import os
import requests
import json
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, sum, mean
import pandas as pd
from matplotlib import pyplot as plt
import hopsworks
import xgboost
import streamlit as st

# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.config(
    "spark.sql.shuffle.partitions", "400").getOrCreate()
sc.setLogLevel('WARN')

# function to fetch raw data from API


def fetch_raw_data(id):
    '''
    params:
        id_list: list of ids for different cities
        api: "https://api.waqi.info/feed/"
        token = b5ffb81511c8f973b9db70d54e76893cc87d2b95
    '''
    api = "https://api.waqi.info/feed/"
    token = "token=b5ffb81511c8f973b9db70d54e76893cc87d2b95"
    url = api + "@" + str(id) + "/?" + token
    instance = requests.get(url)
    if instance.status_code == 200:
       return instance.text
    

arr = [i for i in range(100)]
id_list = sc.parallelize(arr)

raw_data = id_list.map(fetch_raw_data).map(lambda x: json.loads(x))
'''
Cleaning the raw_data to get only the 'data' dictionary for further feature extraction.
'''
cleaned_data = raw_data.map(lambda x: x['data']).filter(
    lambda x: x is not None and len(x.keys()) > 2)

# function definitions to extract features from the (cleaned) raw_data


def get_attr_value(x, a, k):
    if a in x.keys():
        if k in x[a].keys():
            return int(x[a][k]['v'])
    return 0


def get_location(raw_data_rdd):
    location = raw_data_rdd.map(lambda x: (
        x['idx'], x['city']['geo'])).filter(lambda x: x[1] is not None)\
        .map(lambda x: (x[0], x[1][0], x[1][1]))
    return location


def get_aqi(rdd):
    aqi = rdd.map(lambda x: (x['idx'], x['aqi']))
    return aqi


def create_df(rdd, name):
    df = spark.createDataFrame(rdd, ['idx', name])
    return df


# get data
location = get_location(cleaned_data)
aqi = get_aqi(cleaned_data)
main_ft = cleaned_data.map(lambda x: (x['idx'], get_attr_value(x,'iaqi', 'p'),get_attr_value(x,'iaqi', 'h'),\
                                      get_attr_value(x,'iaqi', 't'),get_attr_value(x,'iaqi', 'pm25'),\
                                      get_attr_value(x,'iaqi', 'pm10'), get_attr_value(x,'iaqi', 'o3'),\
                                      get_attr_value(x,'iaqi', 'no2'), get_attr_value(x,'iaqi', 'so2')))

# create dataframes
location_df = spark.createDataFrame(location, ['idx', 'latitude', 'longitude'])
aqi_df = spark.createDataFrame(aqi, ['idx', 'aqi'])
main_ft_df = spark.createDataFrame(main_ft,['idx', 'p','h','t','pm25','pm10','o3','no2','so2'])

# join dataframes
joined_df1= location_df.join(main_ft_df, on='idx', how= 'inner')
joined_df2 = joined_df1.join(aqi_df, on='idx',how='inner')

main_df = joined_df2.toPandas()

def impute_null(main_df, cols):
    for col in cols:
        main_df[col] = main_df[col].fillna(main_df[col].mean())
    return main_df

main_df = impute_null(main_df, list(main_df.columns))

project = hopsworks.login()
fs = project.get_feature_store()

aqi_fg = fs.get_or_create_feature_group(
    name="air_quality_index",
    version="1",
    description="Air Quality Index",
    primary_key=['idx']
)

aqi_fg.insert(main_df)

feature_descriptions = [
    {"name": "idx", "description": "City ID"},
    {"name": "latitude", "description": "Latitude of location"},
    {"name": "longitude", "description": "Longitude of location"},
    {"name": "p", "description": "pressure"},
    {"name": "h", "description": "humidity"},
    {"name": "pm25", "description": "pm2.5"},
    {"name": "pm10", "description": "pm1.0"},
    {"name": "o3", "description": "ozone"},
    {"name": "no2", "description": "nitrogen dioxide"},
    {"name": "so2", "description": "sulphur dioxide"},
    {"name": "aqi", "description": "air quality index"}
    
]
# idx	latitude	longitude	p	h	t	pm25	pm10	o3	no2	so2	aqi
for desc in feature_descriptions: 
    aqi_fg.update_feature_description(desc["name"], desc["description"])
    
ds_query = aqi_fg.select(["idx",	"latitude",	"longitude",	"p",	"h",	"t",    	"pm25",	"pm10",	"o3",	"no2",	"so2",	"aqi"])

try:
    fv_name= 'aqi_view'+str(random.randint(1,1000))
    feature_view = fs.get_feature_view(fv_name, version=1)
except:
    fv_name= 'aqi_view'+str(random.randint(1,1000))
    feature_view = fs.create_feature_view(
        name=fv_name,
        query=ds_query,
        version=1,
        labels=["aqi"]
    )
    
aiq_version, aiq_job = feature_view.create_train_validation_test_split(
    description = 'training set',
    data_format = 'csv',
    validation_size = 0.1,
    test_size = 0.1,
    write_options = {'wait_for_job': True},
    coalesce = True,
)

X_train, X_val, X_test, y_train, y_val, y_test = feature_view.get_train_validation_test_split(1)

# train the model
mod = xgboost.XGBRegressor()
mod = mod.fit(X_train, y_train)

pred = mod.predict(X_val)

st.title('AQI Prediction')
def reflect_action(action):
    if action == "Show dataframe":
        st.write(ds_query.show(5))
    elif action == "SHow columns":
        st.write(main_df.columns)
    elif action == "Show metadata":
        st.write(main_df.describe())

action = st.sidebar.selectbox("Select action", ("Show dataframe","Show columns","Show metadata"))
reflect_action(action)
