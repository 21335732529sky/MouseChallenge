from data_processor import DataProcessor, generate_batch
from model import Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import argparse
from tqdm import tqdm

def predict(model, test_data):
	model.eval()
	preds = []
	labels = []
	for bx, _ in tqdm(generate_batch(test_data, [0,]*len(test_data), shuffle=False, batch_size=1), total=len(test_data)):
		prediction = model.predict(bx).argmax(dim=-1)
		if prediction.is_cuda:
			prediction = prediction.cpu()
		preds.extend(prediction.numpy())

	return preds


def main(args):
	model = Model(
		num_hidden=args.num_hidden,
		num_features=6,
		num_layers=args.num_layers,
		num_labels=2,
		dropout=args.dropout)

	model.load_state_dict(torch.load(args.model_path))
	dp = DataProcessor(
		train_dir=args.train_dir,
		test_dir=args.test_dir)
	test_data = dp.get_test_data()
	with torch.no_grad():
		predictions = predict(model, test_data)
	with open(args.output_file, 'w') as f:
		[f.write(str(p) + '\n') for p in predictions]

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dir', type=str, default='train/')
	parser.add_argument('--test_dir', type=str, default='test/')
	parser.add_argument('--num_hidden', type=int, default=32)
	parser.add_argument('--num_layers', type=int, default=1)
	parser.add_argument('--num_epochs', type=int, default=5)
	parser.add_argument('--dropout', type=float, default=.0)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--model_path', type=str, default='model.bin')
	parser.add_argument('--output_file', type=str, default='test_prediction.csv')

	args = parser.parse_args()
	main(args)