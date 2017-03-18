import numpy as np
import xgboost as xgb
import pdb

def load_data():
	train_feats = open('./train_feats.csv', 'r')
	test_feats = open('./test_feats.csv', 'r')
	X_train = np.random.rand(519164, 512)
	y_train = np.zeros(519164)
	X_test = np.random.rand(129795, 512)
	class_map = {}
	class_map['articulated_truck'] = 0
	class_map['background'] = 1
	class_map['bicycle'] = 2
	class_map['bus'] = 3
	class_map['car'] = 4
	class_map['motorcycle'] = 5
	class_map['non-motorized_vehicle'] = 6
	class_map['pedestrian'] = 7
	class_map['pickup_truck'] = 8
	class_map['single_unit_truck'] = 9
	class_map['work_van'] = 10
	class_map_rev = {}
	class_map_rev[0] = 'articulated_truck'
	class_map_rev[1] = 'background'
	class_map_rev[2] = 'bicycle'
	class_map_rev[3] = 'bus'
	class_map_rev[4] = 'car'
	class_map_rev[5] = 'motorcycle'
	class_map_rev[6] = 'non-motorized_vehicle'
	class_map_rev[7] = 'pedestrian'
	class_map_rev[8] = 'pickup_truck'
	class_map_rev[9] = 'single_unit_truck'
	class_map_rev[10] = 'work_van'
	for i, line in enumerate(train_feats):
		line1 = line.rstrip()
		file_name = line1.split(",")[0]
		feats = line1.split(",")[1:-1]
		#print len(feats)
		feats = np.array(map(lambda x:float(x), feats))
		class_name, file_name = file_name.split("/")[-2], file_name.split("/")[-1]
		X_train[i] = feats
		y_train[i] = class_map[class_name]

	for i, line in enumerate(test_feats):
		line1 = line.rstrip()
		file_name = line1.split(",")[0]
		feats = line.split(",")[1:-1]
		feats = np.array(map(lambda x:float(x), feats))
		file_name = file_name.split("/")[-1]
		X_test[i] = feats

	return X_train, y_train, X_test


X_train, y_train, X_test = load_data()
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pdb.set_trace()

pdb.set_trace()

