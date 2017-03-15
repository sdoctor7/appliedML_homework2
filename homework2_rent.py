## import libraries
import csv
from urllib.request import urlopen
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, Imputer, PolynomialFeatures

def get_rent_data():

	## download and read data into a numpy array
	url = 'https://ndownloader.figshare.com/files/7586326'
	with urlopen(url) as response:
		lines = response.readlines()
	data = np.genfromtxt(lines, delimiter=",", names=True, dtype=None)

	return data

def clean_rent_data(data):

	## convert to pandas dataframe
	data2 = pd.DataFrame(data)

	## remove columns that are specific to current tenant or would 
	## cause information leakage (e.g. rent variables, mortgage value, etc.), 
	## as well as duplicate variables
	data2 = data2.drop(['recid', 'hhr2', 'uf43', 'hhr5', 'race1', 'uf2a', 
		'uf2b', 'sc51', 'sc52', 'sc53', 'sc54', 'sc110', 'sc111', 'sc112', 
		'sc113', 'sc117', 'sc140', 'sc184', 'sc542', 'sc543', 'sc544', 'sc548', 
		'sc549', 'sc550', 'sc551', 'sc570', 'sc574', 'sc560', 'uf53', 'uf54', 
		'new_csr', 'rec1', 'uf46', 'rec4', 'rec_race_a', 'rec_race_c', 
		'tot_per', 'rec28', 'rec39', 'uf42', 'uf42a', 'uf34', 'uf34a', 'uf35', 
		'uf35a', 'uf36', 'uf36a', 'uf37', 'uf37a', 'uf38', 'uf38a', 'uf39',
        'uf39a', 'uf40', 'uf40a', 'uf30', 'uf29', 'rec8', 'rec7', 'fw', 'chufw', 
        'flg_sx1', 'flg_ag1', 'flg_hs1', 'flg_rc1', 'hflag2', 'hflag1', 
        'hflag18', 'uf52h_h', 'uf52h_a', 'uf52h_b', 'uf52h_c', 'uf52h_d', 
        'uf52h_e', 'uf52h_f', 'uf52h_g', 'sc115', 'sc116', 'uf5', 'sc125', 
        'sc143', 'sc174', 'uf64', 'uf17', 'uf17a', 'sc181', 'sc541', 'sc27', 
        'sc152', 'sc153', 'sc155', 'sc156', 'uf26', 'uf28', 'uf27', 'seqno', 
        'hflag13', 'hflag6', 'hflag3', 'hflag14', 'hflag16', 'hflag7', 'hflag9', 
        'hflag10', 'hflag91', 'hflag11', 'hflag12', 'hflag4', 'uf6', 'uf7'], 
        axis=1)

	## divide continuous and categorical variables for processing
	continuous_names = ['sc134', 'uf7a', 'uf8', 'sc150', 'sc151', 'uf12', 
		'uf13', 'uf14', 'uf15', 'uf16']
	continuous = data2[continuous_names]
	categorical = data2
	categorical = categorical.drop(continuous_names, axis=1)

	## drop columns with >50% missing data
	continuous_new = continuous.drop(['sc134', 'uf7a', 'uf8', 'uf13', 'uf14', 
		'uf15', 'uf16'], axis=1)
	categorical_new = categorical.drop(['sc118', 'sc120', 'sc121', 'sc127', 
		'uf9', 'sc141', 'sc144', 'uf10', 'sc173', 'sc193'], axis=1)

	## replace missing values with 'NaN' (varies based on feature)
	subset = categorical_new.ix[:,'uf1_1':'sc38']
	for col in range(len(subset.columns)):
	    column = subset[subset.columns[col]]
	    categorical_new[subset.columns[col]].replace(to_replace=8, value=np.nan, 
	    	inplace=True)
	categorical_new['sc114'].replace(to_replace=4, value=np.nan, inplace=True)
	categorical_new['sc147'].replace(to_replace=[3,8], value=np.nan, inplace=True)
	categorical_new['sc171'].replace(to_replace=[3,8], value=np.nan, inplace=True)
	categorical_new['sc154'].replace(to_replace=[8,9], value=np.nan, inplace=True)
	categorical_new['sc157'].replace(to_replace=[8,9], value=np.nan, inplace=True)
	continuous_new['uf12'].replace(to_replace=9999, value=np.nan, inplace=True)
	categorical_new['sc197'].replace(to_replace=[4,8], value=np.nan, inplace=True)
	categorical_new['sc198'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc187'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc188'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc571'].replace(to_replace=[5,8], value=np.nan, inplace=True)
	categorical_new['sc189'].replace(to_replace=[5,8], value=np.nan, inplace=True)
	categorical_new['sc190'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc191'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc192'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc194'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc196'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc199'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['sc575'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['rec15'].replace(to_replace=[8,10,12], value=np.nan, inplace=True)
	categorical_new['rec21'].replace(to_replace=[8], value=np.nan, inplace=True)
	categorical_new['rec54'].replace(to_replace=[7], value=np.nan, inplace=True)
	categorical_new['rec53'].replace(to_replace=[7], value=np.nan, inplace=True)

	## bring back response variable as y
	y = pd.DataFrame(data)['uf17']

	## find rows in which response variable is unknown and remove them from
	## training and test data; bring continuous and categorical features
	## back together
	indices = [i for i, x in enumerate(y) if x != 99999]
	y2 = y.loc[indices]
	categorical_new2 = categorical_new.loc[indices]
	continuous_new2 = continuous_new.loc[indices]
	data3 = pd.concat([continuous_new2, categorical_new2], axis=1)

	## split train and test data
	X_train, X_test, y_train, y_test = train_test_split(data3, y2, 
		random_state=0)
	colnames = list(X_train.columns.values)

	## impute missing data using knn with 'boro' feature (index 3)
	## knn is trained on training data, then used on both train and test sets
	X_train_knn = np.array(X_train)
	X_test_knn = np.array(X_test)
	for col in range(4,X_train_knn.shape[1]):
	    missing_train = np.isnan(X_train_knn[:,col])
	    missing_test = np.isnan(X_test_knn[:,col])
	    if sum(missing_train) > 0:
	    	## use classifier for categorical data
	        knn = KNeighborsClassifier().fit(X_train_knn[~missing_train,3]
	        	.reshape(-1,1), X_train_knn[~missing_train, col])
	        X_train_knn[missing_train, col] = knn.predict(X_train_knn[missing_train, 3].reshape(-1,1))
	    if sum(missing_test) > 0:
	        X_test_knn[missing_test, col] = knn.predict(X_train_knn[missing_test, 3].reshape(-1,1))
	for col in range(0,3):
	    missing_train = np.isnan(X_train_knn[:,col])
	    missing_test = np.isnan(X_test_knn[:,col])
	    if sum(missing_train) > 0:
	    	## use regressor for continuous data
	        knn = KNeighborsRegressor().fit(X_train_knn[~missing_train,3]
	        	.reshape(-1,1), X_train_knn[~missing_train, col])
	        X_train_knn[missing_train, col] = knn.predict(X_train_knn[missing_train, 3].reshape(-1,1))
	    if sum(missing_test) > 0:
	        X_test_knn[missing_test, col] = knn.predict(X_train_knn[missing_test, 3].reshape(-1,1))

	## scale continuous data using standard scaler
	## (subtract mean, divide by standard deviation)
	## scaler is fit to training data and then used on train and test sets
	scaler = StandardScaler()
	X_train_scaled = X_train_knn.copy()
	X_test_scaled = X_test_knn.copy()
	for col in range(0,3):
	    X_train_scaled[:,col] = scaler.fit_transform(X_train_knn[:,col])
	    X_test_scaled[:,col] = scaler.transform(X_test_knn[:,col])

	## convert back to pandas dataframes, with colnames as before
	X_train_scaled_df = pd.DataFrame(X_train_scaled)
	X_test_scaled_df = pd.DataFrame(X_test_scaled)
	X_train_scaled_df.columns = colnames
	X_test_scaled_df.columns = colnames

	## create dummy variables for categorical variables
	X_train_dummies = pd.get_dummies(X_train_scaled_df, 
		columns=['boro', 'uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5', 'uf1_6', 
		'uf1_7', 'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 'uf1_12', 'uf1_13', 
		'uf1_14', 'uf1_15', 'uf1_16', 'uf1_35', 'uf1_17', 'uf1_18', 'uf1_19', 
		'uf1_20', 'uf1_21', 'uf1_22', 'sc23', 'sc24', 'sc36', 'sc37', 'sc38', 
		'sc114', 'uf48', 'sc147', 'uf11', 'sc149', 'sc171', 'sc154', 'sc157', 
		'sc158', 'sc159', 'sc161', 'sc164', 'sc166', 'sc185', 'sc186', 'sc197', 
		'sc198', 'sc187', 'sc188', 'sc571', 'sc189', 'sc190', 'sc191', 'sc192', 
		'sc194', 'sc196', 'sc199', 'sc575', 'uf19', 'rec15', 'sc26', 'uf23', 
		'rec21', 'rec62', 'rec64', 'rec54', 'rec53', 'cd'])
	X_test_dummies = pd.get_dummies(X_test_scaled_df, 
		columns=['boro', 'uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5', 'uf1_6', 
		'uf1_7', 'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 'uf1_12', 'uf1_13', 
		'uf1_14', 'uf1_15', 'uf1_16', 'uf1_35', 'uf1_17', 'uf1_18', 'uf1_19', 
		'uf1_20', 'uf1_21', 'uf1_22', 'sc23', 'sc24', 'sc36', 'sc37', 'sc38', 
		'sc114', 'uf48', 'sc147', 'uf11', 'sc149', 'sc171', 'sc154', 'sc157', 
		'sc158', 'sc159', 'sc161', 'sc164', 'sc166', 'sc185', 'sc186', 'sc197', 
		'sc198', 'sc187', 'sc188', 'sc571', 'sc189', 'sc190', 'sc191', 'sc192', 
		'sc194', 'sc196', 'sc199', 'sc575', 'uf19', 'rec15', 'sc26', 'uf23', 
		'rec21', 'rec62', 'rec64', 'rec54', 'rec53', 'cd'])

	X_train_array = np.array(X_train_dummies)
	X_test_array = np.array(X_test_dummies)

	return X_train_array, X_test_array, y_train, y_test

def model_rent(X_train, y_train):

	## fit lasso model with polynomial features
	lasso = make_pipeline(PolynomialFeatures(include_bias=False), 
		linear_model.Lasso(alpha=0.4, max_iter=2500))
	lasso.fit(X_train,y_train)

	return lasso	

def score_rent(lasso, X_test, y_test):
	
	## score lasso model
	score = lasso.score(X_test, y_test)

	return score

def predict_rent(lasso, X_test, y_test):

	## predict rent using lasso model
	y_pred = lasso.predict(X_test)

	return X_test, y_test, y_pred
	

