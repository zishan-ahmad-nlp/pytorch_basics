import pandas as pd
import numpy as np
import logging
import fastText
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import json
import math
from keras.utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score,classification_report,precision_recall_fscore_support

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


use_gpu = torch.cuda.is_available()
torch.manual_seed(11)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
if use_gpu:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(11)
    torch.cuda.manual_seed_all(11)

# Here we define our model as a class
class CNN(nn.Module):

	def __init__(self, input_dim, hidden_dim,  emb_weights, vocab_size, output_dim=2,
	num_layers=2):
		super(CNN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.output_dim = output_dim
		emb_dim = 300

		self.embedding = nn.Embedding(vocab_size, emb_dim)
		self.embedding.weight = nn.Parameter(emb_weights, requires_grad=True)
		
		# Define the CNN layer
		self.cnn = nn.Conv2d(1, 100, (2, 300))

		# Define the output layer
		self.linear = nn.Linear(100, self.output_dim)
		self.init_hidden()

	def init_hidden(self):
	#This is what we'll initialise our hidden state as
		torch.nn.init.xavier_uniform_(self.linear.weight.data,np.sqrt(6/(100+self.output_dim)))

		for n, p in self.cnn.named_parameters():
			if 'weight' in n:
				torch.nn.init.orthogonal_(p)
			elif 'bias' in n:
				p.data.zero_()
	



	def forward(self, input_sentence,batch_size):
		# Forward pass through CNN layer
		
		e = self.embedding(input_sentence)
		m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		# print(e.size())
		e = torch.squeeze(e,dim=2)
		# print(e.size())
		e = m(e)
		# print(e.size())
		#input = input.permute(1, 0, 2)
		#lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
		e = e.unsqueeze(1)
		# print(e.size())
		c_out = self.cnn(e)
		c_out = c_out.squeeze(3)
		c_out = F.relu(c_out)
		# print(c_out.size())
		
		c_out = c_out.contiguous().transpose(1,2)
		y_pred = self.linear(c_out)
		y_pred = y_pred.view(batch_size,-1,self.output_dim)
		
		return y_pred




print("loading embeddings...")
ft_hi = fastText.load_model("/home1/zishan/WordEmbeddings/FastText/wiki.bn.bin")

print("loading dictionaries...")
class_index = json.load(open("../../Data/Crosslingual/class_index.json"))
word_index = pd.read_pickle("../../Data/Crosslingual/universal_word_index.pickle")

print("loading training data...")
train = pd.read_pickle('../../Data/Crosslingual/Bengali_train.pickle')
print("loading testing data...")
test = pd.read_pickle('../../Data/Crosslingual/Bengali_test.pickle')
test_trig = np.asarray(test['trigger'].tolist())
test_trig = test_trig.reshape(len(test_trig),75,1)
#test_trig = to_categorical(test_trig,2)
test_sentences = np.asarray(test['sentences_token'].tolist())

vocab_size=len(word_index.items())+1
emb_dim = 300
count = 0
print("creating embedding matrix...")

embedding_matrix = np.zeros((vocab_size, emb_dim),dtype=np.float32)
for i, j in word_index.items():
	word,lang = i
	index, _ = j
	# print(lang)
	try:
		embedding_vector = ft_hi.get_word_vector(word.lower())
			# print(lang)
	except:
		embedding_vector = np.zeros(emb_dim)
		count = count + 1
	embedding_matrix[index] = embedding_vector
print("oov-->",count)

def data_generate(data_name, batch_size, seq_len):
		data_len = len(data_name)
		count = 0
		while True:			
			batch_features_ft = np.zeros((batch_size,seq_len,1),dtype=int)
			batch_triggers = np.zeros((batch_size,seq_len,1),dtype=int)
			batch_triggers_class = np.zeros((batch_size,seq_len,label_size),dtype=int)
			for i in range(batch_size):
				count = count % data_len
				batch = np.asarray(data_name['sentences_token'].loc[count])
				batch_features_ft[i] = batch.reshape(len(batch),1)
				triggers  = np.asarray(data_name['trigger'].loc[count])
				triggers = triggers.reshape(len(triggers),1)
				# triggers = to_categorical(triggers,2)
				batch_triggers[i] = triggers
				triggers_classes = np.asarray(data_name['trigger_class'].loc[count])
				triggers_classes = to_categorical(triggers_classes,label_size)
				batch_triggers_class[i] = triggers_classes
				count = count + 1
			yield batch_features_ft,batch_triggers

def calc_f1_score_trig(real,predict):
		trig_predict = np.argmax(predict.reshape(len(predict)*seq_len,2),axis=-1)
		trig_real = real.reshape(len(real)*seq_len,1)
		return precision_score(trig_real,trig_predict),recall_score(trig_real,trig_predict),f1_score(trig_real,trig_predict)

print("setting parameters...")

seq_len = 75
emb_dim = 300 #ft_hi.get_dimension()
label_size=len(class_index)+1
hidden_dim = 75
learning_rate = 1e-2
batch_size = 32
num_epochs = 40
data_len = len(train)
steps_per_epoch = math.ceil(data_len/batch_size)
emb_weights = torch.from_numpy(embedding_matrix).cuda()
print(emb_weights.data[10])
# model = LSTM(emb_dim, 75, output_dim=2, num_layers=2, vocab_size=vocab_size,emb_weights=emb_weights)
model = CNN(emb_dim, 75, output_dim=2, num_layers=2, vocab_size=vocab_size,emb_weights=emb_weights)

model = model.cuda()
# loss_fn = torch.nn.CrossEntropyLoss(reduce = True)
loss_fn = torch.nn.CrossEntropyLoss(reduce = True)

print([i.requires_grad for i in model.parameters()])
optimiser = torch.optim.Adam([i for i in model.parameters() if i.requires_grad], lr=learning_rate)


#####################
# Train model
#####################
X_train = np.asarray(train['sentences_token'].tolist())
y_train = np.asarray(train['trigger'].tolist())


count = 1
# hist = []*num_epochs
for t in range(num_epochs):
    # Clear stored gradient
	print("Epoch-->",count)
	c = 0
	loss = 0
	model.zero_grad()
	for x,y in data_generate(train,batch_size,seq_len):
		if c == steps_per_epoch:
			break
		c = c + 1
		# Initialise hidden state
		# Don't do this if you want your LSTM to be stateful
		torch.set_grad_enabled(True)
		x_train = torch.from_numpy(x).cuda()
		y_pred = model(x_train,batch_size=batch_size)
		#print(y_pred.size())
		y_train = torch.from_numpy(y).cuda()
		#print(y.shape)
		loss = loss_fn(y_pred.view(-1,2), y_train.view(-1))
		loss.backward()
		optimiser.step()
		optimiser.zero_grad()

	# Zero out gradient, else they will accumulate between epochs
	count = count + 1
	# Backward pass
	print("end epoch")
	torch.set_grad_enabled(False)
	print(test_sentences.shape)
	x_test = torch.from_numpy(test_sentences).cuda()
	y_test_pred = model(x_test,batch_size=len(test_sentences))
	# y_test_pred = torch.nn.functional.softmax(y_test_pred)
	y_test_pred = torch.nn.Softmax(dim=-1)(y_test_pred)
	print(y_test_pred.size())

	# y_test_pred = torch.nn.Softmax(dim=-1)(y_test_pred)

	p,r,f = calc_f1_score_trig(test_trig,y_test_pred.cpu().numpy())
	print("Epoch ", t+1, "Loss: ", loss.item(),"Precision:",p,"Recall:",r,"F-Score:",f)
	# hist[t] = loss.item()

	# Update parameters
