# data set
# D = {(x1, y1, (x2, y2) ... (xN, yN)}
# where xi = (xi1, xi2, ..., xin), n is number of features
# and yi = {1, 2, ..., K}, i = 1, 2, ..., N, N is number of samples

# X is a discrete random sample
# P(X = xi) = pi, i = 1, 2, ..., n
# the entropy of the random variable is 
# H(X) = - ∑ pi ln pi


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

import pprint
from numpy import log2

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

datasets, labels = create_data()

train_data = pd.DataFrame(datasets, columns = labels)

# entropy
def calc_ent(datasets):
	data_length = len(datasets)
	label_count = {}
	for i in range(data_length):
		label = datasets[i][-1]
		if label not in label_count:
			label_count[label] = 0
		label_count[label] += 1

	ent = -sum([(p / data_length) * log2(p / data_length) 
		for p in label_count.values()]) 
	return ent

# conditional entropy
def cond_ent(datasets, axis = 0):
	
	data_length = len(datasets)
	features_sets = {}

	for i in range(data_length):
		feature = datasets[i][axis]
		if feature not in features_sets:
			features_sets[feature] = []
		features_sets[feature].append(datasets[i])
	cond_ent = sum(
		[len(p)/data_length * calc_ent(p) for p in features_sets.values()])

	return cond_ent

# information gain
def info_gain(ent, cond_ent):
	return ent - cond_ent

def info_gain_train(datasets):

	count = len(datasets[0]) - 1
	ent = calc_ent(datasets)

	best_feature = []
	for c in range(count):
		c_info_gain = info_gain(ent, cond_ent(datasets, axis = c))
		best_feature.append((c, c_info_gain))
		print('特征({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))

	best_ = max(best_feature, key = lambda x: x[-1])
	return '特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]])


class Node:

	def __init__(self, root = True, label = None, feature_name = None, feature=None):
		self.root = root
		self.label = label
		self.feature_name = feature_name
		self.feature = feature
		self.tree = {}
		self.result = {
		'label': self.label,
		'feature': self.feature,
		'tree':self.tree
		}

	def __repr__(self):
		return '{}'.format(self.result)

	def add_node(self, val, node):
		self.tree[val] = node

	def predict(self, features):
		if self.root is True:
			return self.label
		return self.tree[features[self.feature]].predict(features)

class DTree:

	def __init__(self, epsilon = 0.1):
		self.epsilon = epsilon
		self._tree = {}

	@staticmethod
	def calc_ent(datasets):
		data_length = len(datasets)
		label_count = {}
		for i in range(data_length):
			label = datasets[i][-1]
			if label not in label_count:
				label_count[label] = 0
			label_count[label] += 1

		ent = -sum([(p / data_length) * log2(p / data_length) 
			for p in label_count.values()]) 
	return ent

	def cond_ent(datasets, axis = 0):
	
		data_length = len(datasets)
		features_sets = {}

		for i in range(data_length):
			feature = datasets[i][axis]
			if feature not in features_sets:
			featureeatures_sets[feature] = []
			features_sets[feature].append(datasets[i])
		cond_ent = sum(
			[len(p)/data_length * calc_ent(p) for p in features_sets.values()])

		return cond_ent

	# information gain
	@staticmethod
	def info_gain(ent, cond_ent):
		return ent - cond_ent

	def info_gain_train(self, datasets):

		count = len(datasets[0]) - 1
		ent = calc_ent(datasets)

		best_feature = []
		for c in range(count):
			c_info_gain = info_gain(ent, cond_ent(datasets, axis = c))
			best_feature.append((c, c_info_gain))
		

		best_ = max(best_feature, key = lambda x: x[-1])
		return best_

	def train(self, train_data):
		








if __name__ == "__main__":
	info_gain_train(np.array(datasets))






























