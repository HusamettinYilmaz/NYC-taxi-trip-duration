import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from haversine import haversine


train_data = pd.read_csv("/home/husammm/Desktop/Courses/ML/Projects/1 project-nyc-taxi-trip-duration/split/train.csv")
val_data = pd.read_csv("/home/husammm/Desktop/Courses/ML/Projects/1 project-nyc-taxi-trip-duration/split/val.csv")

pd.set_option('display.max_columns', None)
pd.set_option("display.float_format", "{:.3f}".format)

def calc_distance(data):
    pickup = (data['pickup_latitude'], data['pickup_longitude'])
    drop = (data['dropoff_latitude'], data['dropoff_longitude'])
    return haversine(pickup, drop)

def Ranges(data, ranges):
        # making ranges for hours per day
        # time_ranges = ranges
        bins = []

        for value in data:
            for idx, (min_val, max_val) in enumerate(ranges):
                if min_val <= int(value) < max_val:
                    bins.append(idx)    
        data = bins
        return data

def split_data(data):
    ## Splitting data to Features and Target
    data = data.to_numpy()

    x = data[:, :-1]
    t = data[:, -1:].reshape(-1, 1)
    
    return x, t

def on_target(data):
    ## convert time from seconds to minutes 
    data['trip_duration'] = data['trip_duration'] / 60

    ## deleting data which has trip_duration more than 90 minutes
    data = data[data.trip_duration <= 90]
    
    return data

def calc_speed(data, is_test = True):
    if is_test == True:
        return data
    else:      
        data['speed'] = (data.distance/(data.trip_duration/60))
        
        ## deleting data which has Speed more than 105 KM
        data = data[data.speed <= 105]
        
        return data

def EDA_clean_data(data):
    
    ## deleting data which passenger_count less than 1 or more than 6
    data = data[1 <= data.passenger_count]
    data = data[data.passenger_count <= 6]
    
    # claculate distance
    data['distance'] = data.apply(lambda x: calc_distance(x), axis = 1)
    ####################################################
    # print((data.distance > 100).value_counts())
    # print((data.distance < 0.3).value_counts())

    ## deleting data which has distance more than 100KM and less than 300M
    data = data[0.5 < data.distance]
    data = data[data.distance <= 100]

    ## calculate Speed
    # data = calc_speed(data, is_test= True)
    
    # plt.figure(figsize = (8, 3))
    # sns.boxplot(data.distance)
    # plt.show()

    ## seperate datetime as Hour, Day, Month in diff columns
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    
    data['time'] = data['pickup_datetime'].dt.hour
    data['day'] = data['pickup_datetime'].dt.weekday
    data['month'] = data['pickup_datetime'].dt.month
    # train_data['year'] = train_data['pickup_datetime'].dt.year
    
    ## drop unnecessary columns like id etc...
    data.drop(columns=['id', 'vendor_id', 'pickup_datetime', 'passenger_count', 'store_and_fwd_flag'], axis=1, inplace= True)

    # print(data.columns)

    index = ['pickup_longitude', 'pickup_latitude',
            'dropoff_longitude','dropoff_latitude', 'time', 'day',
            'month', 'distance', 'trip_duration']
    data = data.reindex(columns= index)

    # data['time']=data['time'].astype(str)
    # data['day']=data['day'].astype(str)
    # data['month']=data['month'].astype(str)
    #################################################################
 
    time_ranges = [(0, 5), (5, 9), (9, 12), (12, 16), (16, 19), (19, 24)]
    data['time'] = Ranges(data['time'], time_ranges)
    
    # print(data.describe())
    
    return data

def transformer(x_train, degree):
    ## applaying Polynomial features to data
    poly = PolynomialFeatures(degree= degree, include_bias= False, interaction_only= False)
    x_train = poly.fit_transform(x_train)
    
    
    ## scaling the data
    scale = MinMaxScaler().fit(x_train)
    x_train = MinMaxScaler().fit_transform(x_train)
    
    
    return x_train, scale

def Pickle(file_path, pickle_dct ):
    
    with open (file_path, "wb") as Trip_dur_f:
        pickle.dump(model, Trip_dur_f)

def Model(x_train, t_train):
    ## trying of linear reg.
    """
        # model = LinearRegression(fit_intercept= True)
        # model.fit(x_train, t_train)

        # pred_train = model.predict(x_train)
        # pred_val = model.predict(x_val)

            
        # pred_train = model.predict(x_train)
        # pred_val = model.predict(x_val)

        # mean_train_error = mean_squared_error(t_train, pred_train, squared= False)
        # mean_val_error = mean_squared_error(t_val, pred_val, squared= False)

        # r2_train_error = r2_score(t_train, pred_train)
        # r2_val_error = r2_score(t_val, pred_val)
    """    
    ## applaying Ridge
    model = Ridge(alpha= 1)
    model.fit(x_train, t_train)

    return model


if __name__ == "__main__":

    """
        # steps = (
        #     ('EDA_clean_data', EDA_clean_data),
        #     ('Split_data', split_data),
        #     ('Model', Model)
        # )

        # pipeline = Pipeline(steps)
        # # pipeline.fit(x_train, t_train)
        # train_data = pipeline.EDA_clean_data(train_data)
        # x_train, t_train = pipeline.Split_data(train_data)
        # model, mean_train_error, r2_train_error = pipeline.Model(x_train, t_train)

        # val_data = pipeline.EDA_clean_data(val_data)
        # x_val, t_val = pipeline.Split_data(val_data)
        # model, mean_val_error, r2_val_error = pipeline.Model(x_val, t_val)
    """
    train_data = on_target(train_data)
    val_data = on_target(val_data)
    
    train_data = EDA_clean_data(train_data)
    val_data = EDA_clean_data(val_data)

    x_train, t_train = split_data(train_data)
    x_val, t_val = split_data(val_data)

    x_train, scale = transformer(x_train, degree= 3)
    poly = PolynomialFeatures(degree= 3, include_bias= False, interaction_only= False)
    x_val = poly.fit_transform(x_val)
    x_val = scale.transform(x_val)

    model = Model(x_train, t_train)

    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)

    mean_train_error = mean_squared_error(t_train, pred_train, squared= False)
    mean_val_error = mean_squared_error(t_val, pred_val, squared= False)

    r2_train_error = r2_score(t_train, pred_train)
    r2_val_error = r2_score(t_val, pred_val)

    print(f"Mean Train error is:{mean_train_error}")
    print(f"Mean Val error is:{mean_val_error}")
    print(f"R2 Train error is:{r2_train_error}")
    print(f"R2 Val error is:{r2_val_error}")

    ...

    # pickle_dct = {"On_target" : on_target,
    #               "EDA_clean_data" : EDA_clean_data,
    #               "Split_data" : split_data,
    #               "Transformer" : transformer, "Model" : Model}

    file_path = "/home/husammm/Desktop/Courses/Python/MLCourse/Trip_duration_prediction/Trip_duration_predection.pkl"
    Pickle(file_path, model)

