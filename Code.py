# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 20:54:45 2018

@author: subhajit
"""
#from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize']=(10,18)
#%matplotlib inline
from datetime import datetime
from datetime import date
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns # plot beautiful charts
import warnings
sns.set()
#warnings.filterwarnings('ignore')


data  = pd.read_csv('train.csv')

data.head(3)

data.info()

data.pickup_datetime=pd.to_datetime(data.pickup_datetime)
data.dropoff_datetime=pd.to_datetime(data.dropoff_datetime)

test  = pd.read_csv('test.csv')

test.info()

test.pickup_datetime=pd.to_datetime(test.pickup_datetime)

for df in (data,test):
    df['year']  = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day']   = df['pickup_datetime'].dt.day
    df['hr']    = df['pickup_datetime'].dt.hour
    df['minute']= df['pickup_datetime'].dt.minute
    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
    
test.head(3)

holiday = pd.read_csv('NYC_2016Holidays.csv',sep=';')

holiday['Date'] = holiday['Date'].apply(lambda x: x + ' 2016')

holidays = [datetime.strptime(holiday.loc[i,'Date'], '%B %d %Y').date() for i in range(len(holiday))]

time_data = pd.DataFrame(index = range(len(data)))
time_test = pd.DataFrame(index = range(len(test)))

#If there is an outlier in trip_duration (as we will see), the transform help mitigate the effect. 
#If the value is still enormously big after the log transform, we can tell it is a true outliers with more confidence.

data = data.assign(log_trip_duration = np.log(data.trip_duration+1))

# Pickup Time and Weekend Features

from datetime import date
def restday(yr,month,day,holidays):
    '''
    Output:
        is_rest: a list of Boolean variable indicating if the sample occurred in the rest day.
        is_weekend: a list of Boolean variable indicating if the sample occurred in the weekend.
    '''
    is_rest    = [None]*len(yr)
    is_weekend = [None]*len(yr)
    i=0
    for yy,mm,dd in zip(yr,month,day):        
        is_weekend[i] = date(yy,mm,dd).isoweekday() in (6,7)
        is_rest[i]    = is_weekend[i] or date(yy,mm,dd) in holidays 
        i+=1
    return is_rest,is_weekend

rest_day,weekend = restday(data.year,data.month,data.day,holidays)
time_data = time_data.assign(rest_day=rest_day)
time_data = time_data.assign(weekend=weekend)

rest_day,weekend = restday(test.year,test.month,test.day,holidays)
time_test = time_test.assign(rest_day=rest_day)
time_test = time_test.assign(weekend=weekend)

time_data = time_data.assign(pickup_time = data.hr+data.minute/60)#float value. E.g. 7.5 means 7:30 am
time_test = time_test.assign(pickup_time = test.hr+test.minute/60)


time_data.to_csv('featurestime_data.csv',index=False)
time_test.to_csv('featurestime_test.csv',index=False)

# Distance Features

#1.2.1 OSRM Features
#
#The first thing comes to mind is using the GPS location to aquire the real distance between pickup point and dropoff one,
#as you will see in Subsection Other Distance Features.
#However, travel distance should be more relevent here. The difficault part is to aquire this feature.
#Thanks to Oscarleo who manage to pull it off from OSRM. Let's put together the most relative columns.
#    

fastrout1 = pd.read_csv('fastest_routes_train_part_1.csv',
                        usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps','step_direction'])
fastrout2 = pd.read_csv('fastest_routes_train_part_2.csv',
                        usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps','step_direction'])
fastrout = pd.concat((fastrout1,fastrout2))
fastrout.head()

#Generate a feature for number of right turns and left turns.

right_turn = []
left_turn = []
right_turn+= list(map(lambda x:x.count('right')-x.count('slight right'),fastrout.step_direction))
left_turn += list(map(lambda x:x.count('left')-x.count('slight left'),fastrout.step_direction))

osrm_data = fastrout[['id','total_distance','total_travel_time','number_of_steps']]
osrm_data = osrm_data.assign(right_steps=right_turn)
osrm_data = osrm_data.assign(left_steps=left_turn)
osrm_data.head(3)

data = data.join(osrm_data.set_index('id'), on='id')
data.head(3)

osrm_test = pd.read_csv('fastest_routes_test.csv')
right_turn= list(map(lambda x:x.count('right')-x.count('slight right'),osrm_test.step_direction))
left_turn = list(map(lambda x:x.count('left')-x.count('slight left'),osrm_test.step_direction))

osrm_test = osrm_test[['id','total_distance','total_travel_time','number_of_steps']]
osrm_test = osrm_test.assign(right_steps=right_turn)
osrm_test = osrm_test.assign(left_steps=left_turn)
osrm_test.head(3)

test = test.join(osrm_test.set_index('id'), on='id')

osrm_test.head()

osrm_data = data[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]
osrm_test = test[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]

data.to_csv('featuresosrm_data.csv',index=False,
            columns = ['total_distance','total_travel_time','number_of_steps','right_steps','left_steps'])
test.to_csv('featuresosrm_test.csv',index=False,
            columns = ['total_distance','total_travel_time','number_of_steps','right_steps','left_steps'])

#Other Distance Features
#
#Haversine distance: the direct distance of two GPS location, taking into account that the earth is round.
#Manhattan distance: the usual L1 distance, here the haversine distance is used to calculate each coordinate of distance.
#Bearing: The direction of the trip. Using radian as unit. (I must admit that I am not fully understand the formula. I have starring at it for a long time but can't come up anything. If anyone can help explain that will do me a big favor.)


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


List_dist = []
for df in (data,test):
    lat1, lng1, lat2, lng2 = (df['pickup_latitude'].values, df['pickup_longitude'].values, 
                              df['dropoff_latitude'].values,df['dropoff_longitude'].values)
    dist = pd.DataFrame(index=range(len(df)))
    dist = dist.assign(haversind_dist = haversine_array(lat1, lng1, lat2, lng2))
    dist = dist.assign(manhattan_dist = dummy_manhattan_distance(lat1, lng1, lat2, lng2))
    dist = dist.assign(bearing = bearing_array(lat1, lng1, lat2, lng2))
    List_dist.append(dist)
Other_dist_data,Other_dist_test = List_dist

Other_dist_data.to_csv('featuresOther_dist_data.csv')
Other_dist_test.to_csv('featuresOther_dist_test.csv')


#Location Features: K-means Clustering

#Besides of keeping entire list of latitude and longitute, I would like to distinguish them with few approximately locations. 
#It might be helpful our xgboost since it will take 4 branches of tree (remember I will use a tree-based algorithm) to specify a square region. 
#The best way, I guess, is to separate these points by the town of the location, which requires some geological information from google API or OSRM. Besides OSRM, you can acquire googlemap information from Googlemap API.
#
#I setup 10 kmeans clusters for our data set.
# The kmeans method is performed on 4-d data ['pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude']
#
#Note that the resulting feature is categorical value between 0,1,2,...,10. To use this feature in xgboost, you have to transform it to 20 dummy variables, otherwise the module will treat it as numerical.


coord_pickup = np.vstack((data[['pickup_latitude', 'pickup_longitude']].values,                  
                          test[['pickup_latitude', 'pickup_longitude']].values))
coord_dropoff = np.vstack((data[['dropoff_latitude', 'dropoff_longitude']].values,                  
                           test[['dropoff_latitude', 'dropoff_longitude']].values))

coords = np.hstack((coord_pickup,coord_dropoff))# 4 dimensional data
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10000).fit(coords[sample_ind])
for df in (data,test):
    df.loc[:, 'pickup_dropoff_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude',
                                                         'dropoff_latitude','dropoff_longitude']])
    
kmean10_data = data[['pickup_dropoff_loc']]
kmean10_test = test[['pickup_dropoff_loc']]    

data.to_csv('featureskmean10_data.csv',index=False,columns = ['pickup_dropoff_loc'])
test.to_csv('featureskmean10_test.csv',index=False,columns = ['pickup_dropoff_loc'])

#Let's take a look at our kmeans clusters.
# Remember that each cluster is represented by a 4d point (longitude/latitude of pickup location and logitude/latitude of dropoff location). 
# I will drow first 500 points of each clusters, and each plot have two locations, pickup and dropoff.

plt.figure(figsize=(16,16))
N = 500
for i in range(10):
    plt.subplot(4,3,i+1)
    tmp_data = data[data.pickup_dropoff_loc==i]
    drop = plt.scatter(tmp_data['dropoff_longitude'][:N], tmp_data['dropoff_latitude'][:N], s=10, lw=0, alpha=0.5,label='dropoff')
    pick = plt.scatter(tmp_data['pickup_longitude'][:N], tmp_data['pickup_latitude'][:N], s=10, lw=0, alpha=0.4,label='pickup')    
    plt.xlim([-74.05,-73.75]);plt.ylim([40.6,40.9])
    plt.legend(handles = [pick,drop])
    plt.title('clusters %d'%i)
    
#Some interesting facts:

#    The short trips are clustered to the same clusters.
#    Pickup location is more concentrated than dropoff one. This might just because the taxi driver doesn't want to reach out too far to pickup. Or there are more convenient way (like train) for people to go to the city
    
weather = pd.read_csv('KNYC_Metars.csv', parse_dates=['Time'])
weather.head(3)

print('The Events has values {}.'.format(str(set(weather.Events))))

weather['snow']= 1*(weather.Events=='Snow') + 1*(weather.Events=='Fog\n\t,\nSnow')
weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather['hr'] = weather['Time'].dt.hour
weather = weather[weather['year'] == 2016][['month','day','hr','Temp.','Precip','snow','Visibility']]

weather.head()

data = pd.merge(data, weather, on = ['month', 'day', 'hr'], how = 'left')
test = pd.merge(test, weather, on = ['month', 'day', 'hr'], how = 'left')

weather_data = data[['Temp.','Precip','snow','Visibility']]
weather_test = test[['Temp.','Precip','snow','Visibility']]

weather_data.to_csv('featuresweather_data.csv',index=False)
weather_test.to_csv('featuresweather_test.csv',index=False)

#Outliers..

outliers=np.array([False]*len(data))

print('There are %d rows that have missing values'%sum(data.isnull().any(axis=1)))

# There are 56598 rows that have missing values

# Outliers from 'trip_duration'

#trip_duration too big
#
#Note that for column 'trip_duration', the unit is second. There are some samples that have ridiculus 100 days ride; that's the outliers.
# There are several tools to help you detect them: "run sequence plot", and "boxplot".

y = np.array(data.log_trip_duration)
plt.subplot(131)
plt.plot(range(len(y)),y,'.');plt.ylabel('log_trip_duration');plt.xlabel('index');plt.title('val vs. index')
plt.subplot(132)
sns.boxplot(y=data.log_trip_duration)
plt.subplot(133)
sns.distplot(y,bins=50, color="m");plt.yticks([]);plt.xlabel('log_trip_duration');plt.title('data');plt.ylabel('frequency')

#Looks like there are just 4 points that really far off. Also, there are a lot of trip duration in
#exp(11.37)−1≈86400,
#which is 24 hrs. Well, I think it is still a possible thing, sometimes you take a taxi to a certain place then come back, and the clock just reset for every day, or simply they just forgot to turn off...
# That is the story I imagine. We can not treat them as outliers, despite how outliers they are, since it will happened in test set also. 
#But for the 4 points, I think cast them to outliers is wise.

outliers[y>12]=True
print('There are %d entries that have trip duration too long'% sum(outliers))

#trip_duration too small

kph = osrm_data.total_distance/1000/data.trip_duration*3600
plt.plot(range(len(kph)),kph,'.');plt.ylabel('kph');plt.xlabel('index');plt.show()


#As you can see, there are a lot of taxi drive beyond 1000 km/hr; there is a lot of recording error. If I am predict new values I will remove it.
# For this competition I simply keep all of them. If you still can't tolerant this error, you can use the following code to cast them to outliers.

kph = np.array(kph)
kph[np.isnan(kph)]=0.0
dist = np.array(osrm_data.total_distance) #note this is in meters
dist[np.isnan(dist)]=0.0
outliers_speeding = np.array([False]*len(kph))
outliers_speeding = outliers_speeding | ((kph>300) & (dist>1000))
outliers_speeding = outliers_speeding | ((kph>180)  & (dist<=1000))                                         
outliers_speeding = outliers_speeding | ((kph>100)  & (dist<=100))                                         
outliers_speeding = outliers_speeding | ((kph>80)  & (dist<=20))
print('There are %d speeding outliers'%sum(outliers_speeding))   
outliers = outliers|outliers_speeding


#utliers from Locations

fig=plt.figure(figsize=(10, 8))
for i,loc in enumerate((['pickup_longitude','dropoff_longitude'],['pickup_latitude','dropoff_latitude'])):
    plt.subplot(1,2,i+1)
    sns.boxplot(data=data[outliers==False],order=loc)
    
outliers[data.pickup_longitude<-110]=True
outliers[data.dropoff_longitude<-110]=True
outliers[data.pickup_latitude>45]=True
print('There are total %d entries of ouliers'% sum(outliers))

Out = pd.DataFrame()
Out = Out.assign(outliers=outliers)

data = data.assign(outliers=outliers)

data.to_csv('featuresoutliers.csv',index=False,columns=['outliers'])


#Analysis of Features
#pickup_time vs trip_duration

tmpdata = data
tmpdata = pd.concat([tmpdata,time_data],axis = 1)
tmpdata = tmpdata[outliers==False]

fig=plt.figure(figsize=(18, 8))
sns.boxplot(x="hr", y="log_trip_duration", data=tmpdata)

#month vs trip_duration

sns.violinplot(x="month", y="log_trip_duration", hue="rest_day", data=tmpdata, split=True,inner="quart")

#XGB Model: the Prediction of trip_duration

mydf = data[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag']]

testdf = test[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag']]

outliers = pd.read_csv('featuresoutliers.csv')
time_data       = pd.read_csv('featurestime_data.csv')
weather_data    = pd.read_csv('featuresweather_data.csv')
osrm_data       = pd.read_csv('featuresosrm_data.csv')
Other_dist_data = pd.read_csv('featuresOther_dist_data.csv')
kmean10_data    = pd.read_csv('featureskmean10_data.csv')

time_test       = pd.read_csv('featurestime_test.csv')
weather_test    = pd.read_csv('featuresweather_test.csv')
osrm_test       = pd.read_csv('featuresosrm_test.csv')
Other_dist_test = pd.read_csv('featuresOther_dist_test.csv')
kmean10_test    = pd.read_csv('featureskmean10_test.csv')

#kmean10_data and kmean10_test is categorical, we need to transform it to one-hot representation for xgboost module to treat it correctly.

kmean_data= pd.get_dummies(kmean10_data.pickup_dropoff_loc,prefix='loc', prefix_sep='_')    
kmean_test= pd.get_dummies(kmean10_test.pickup_dropoff_loc,prefix='loc', prefix_sep='_')    

mydf  = pd.concat([mydf  ,time_data,weather_data,osrm_data,Other_dist_data,kmean_data],axis=1)
testdf= pd.concat([testdf,time_test,weather_test,osrm_test,Other_dist_test,kmean_test],axis=1)

if np.all(mydf.keys()==testdf.keys()):
    print('Good! The keys of training feature is identical to those of test feature')
else:
    print('Oops, something is wrong, keys in training and testing are not matching') 
    
X = mydf[Out.outliers==False]
z = data[Out.outliers==False].log_trip_duration.values

if np.all(X.keys()==testdf.keys()):
    print('Good! The keys of training feature is identical to those of test feature.')
    print('They both have %d features, as follows:'%len(X.keys()))
    print(list(X.keys()))
else:
    print('Oops, something is wrong, keys in training and testing are not matching')
    
import xgboost as xgb

X = X[:50000]
z = z[:50000]


from sklearn.model_selection import train_test_split

Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)
Xcv,Xv,Zcv,Zv = train_test_split(Xval, Zval, test_size=0.5, random_state=1)
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xcv   , label=Zcv)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]

parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:linear',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10,
         'nthread'  :3}  #number of cpu core to use

model = xgb.train(parms, data_tr, num_boost_round=1000, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=100)

print('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))

#to get better score, here is my suggestion:
#
#    Using lower eta
#    Using lower min_child_weight
#    Using larger max_depth


#The Submission

data_test = xgb.DMatrix(testdf)
ztest = model.predict(data_test)

#Convert z back to y:
#y=e^z−1

ytest = np.exp(ztest)-1
print(ytest[:10])

submission = pd.DataFrame({'id': test.id, 'trip_duration': ytest})

submission.to_csv('submission.csv', index=False)

#Importance for each Feature

fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(model,ax = axes,height =0.5)
plt.show();
