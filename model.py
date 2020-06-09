import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
	def __init__(self,
				 num_hidden=32,
				 num_features=6,
				 num_layers=1,
				 num_labels=2,
				 dropout=.0):
		super().__init__()
		self.lstm = torch.nn.LSTM(
						input_size=num_features,
						hidden_size=num_hidden,
						num_layers=num_layers,
						batch_first=True,
						dropout=dropout)
		self.cls = torch.nn.Linear(num_hidden, num_labels)
		self.loss = torch.nn.NLLLoss()
		self.log_softmax = torch.nn.LogSoftmax(dim=-1)

	def forward(self, inputs, labels):
		output, _ = self.lstm(inputs)
		output = output[:, -1, :]
		logits = self.cls(output)
		log_logits = self.log_softmax(logits)

		return self.loss(log_logits, labels)

	def predict(self, inputs):
		output, _ = self.lstm(inputs)
		output = output[:, -1, :]
		logits = self.cls(output)
		
		return F.softmax(logits, dim=-1)

