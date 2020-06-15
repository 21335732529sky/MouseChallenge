import pandas as pd
import numpy as np
import random
import os
import torch

class DataProcessor:
	def __init__(self, train_dir, test_dir):
		self.train_dir = train_dir
		self.test_dir = test_dir

	def get_train_data(self):
		train_data, train_labels = self.read_data(self.train_dir)
		train_data, train_labels = self.preprocess(train_data, train_labels)

		return train_data, train_labels

	def get_test_data(self):
		test_data, _ = self.read_data(self.test_dir)

		return test_data

	def read_data(self, train_dir):
		data = []
		label = []

		for i, fname in enumerate(os.listdir(train_dir)):
			df = pd.read_csv(os.path.join(train_dir, fname))
			data.append(df.values[:, 1:]) #フレーム以外の全て
			label.append((0 if i < 5 else 1))

		return data, label

	def preprocess(self, train_data, train_labels):
		train_data, train_labels = self.split_data(train_data, train_labels)

		return train_data, train_labels

	def split_data(self, train_data, train_labels, window=240):
		"""
		読み込んだデータをwindowごとに区切る
		開始位置を (0, 0)にする
		"""
		new_data = []
		new_labels = []
		for data, label in zip(train_data, train_labels):
			num_data = len(data) // window
			sliced_data = data[:num_data*window].reshape((num_data, window, -1))
			sliced_data -= sliced_data[:, 0:1]
			new_data.extend(sliced_data)

			new_labels.extend([label,]*num_data)

		return new_data, new_labels

def generate_batch(inputs, labels, batch_size=32, shuffle=True, cuda=False, size_prop=1.0):
	idx = list(range(len(inputs)))
	if shuffle:
		random.shuffle(idx)
	idx = idx[:int(len(idx) * size_prop)]
	c = 0
	while c < len(idx):
		bi = idx[c:c+batch_size]
		bx = [inputs[i] for i in bi]
		by = [labels[i] for i in bi]
		bx = torch.tensor(bx).type(torch.FloatTensor)
		by = torch.tensor(by)
		if cuda:
			bx = bx.cuda()
			by = by.cuda()
		c += batch_size

		yield bx, by










