import torch
import pandas 

class DataProcessor:
	def __init__(self, file_name):
		df = pd.read_csv(file_name)
		