from data_processor import DataProcessor, generate_batch
from model import Model, EnsembleModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import argparse

def evaluate(model, dataset, cuda=False):
	model.eval()
	preds = []
	labels = []
	for bx, by in generate_batch(*dataset, shuffle=False, cuda=cuda):
		prediction = model.predict(bx).argmax(dim=-1)
		if prediction.is_cuda:
			prediction = prediction.cpu()
			by = by.cpu()
		preds.extend(prediction)
		labels.extend(by)

	score = accuracy_score(labels, preds)
	print('Accuracy: {:.2f}'.format(score * 100))

	model.train()

	return score


def train(model,
		  dataset_train,
		  dataset_dev,
		  lr=1e-3,
		  batch_size=32,
		  num_epochs=5,
		  cuda=False,
      size_prop=1.0):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	steps = 0

	for i in range(num_epochs):
		for bx, by in generate_batch(*dataset_train, batch_size=batch_size, cuda=cuda, size_prop=size_prop):
			optimizer.zero_grad()
			loss = model(bx, by)
			loss.backward()
			optimizer.step()
			steps += 1
			loss_value = loss.detach()
			if loss_value.is_cuda:
				loss_value = loss_value.cpu()
			print('\rstep: {}, loss = {:.5f}'.format(steps, loss_value.numpy()), end='')
		print('')
		evaluate(model, dataset_dev, cuda=cuda)



def main(args):
	dp = DataProcessor(
		train_dir=args.train_dir,
		test_dir=args.test_dir)
	train_data, train_labels = dp.get_train_data()
	train_x, dev_x, train_y, dev_y = train_test_split(train_data,
													  train_labels,
													  test_size=0.2,
													  shuffle=True)
	model = Model(
		num_hidden=args.num_hidden,
		num_features=6,
		num_layers=args.num_layers,
		num_labels=2,
		dropout=args.dropout)
	if args.cuda:
		model.cuda()

	train(model,
		  (train_x, train_y),
		  (dev_x, dev_y),
		  lr=args.lr,
		  num_epochs=args.num_epochs,
		  batch_size=args.batch_size,
		  cuda=args.cuda)

	torch.save(model.state_dict(), args.model_path)

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
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--model_path', type=str, default='model.bin')
	# parser.add_argument('--ensemble', action='store_true')
	# parser.add_argument('--num_models', type=int, default=10)
	# parser.add_argument('--size_prop', type=float, default=1.0)

	args = parser.parse_args()
	main(args)


