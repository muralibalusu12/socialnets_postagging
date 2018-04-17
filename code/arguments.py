import dynet as dy
import argparse
import logging
import os
import random
from collections import Counter, defaultdict
#import matplotlib.pyplot as plt

def get_arguments():
	# Training settings
	parser = argparse.ArgumentParser(description='LSTM Baseline')

	parser.add_argument('--train-data', type=str, default='../data/oct27.train', 
						help='path to the train data file')
	parser.add_argument('--dev-data', type=str, default='../data/oct27.dev', 
						help='path to the dev data file')
	parser.add_argument('--test-data', type=str, default='../data/oct27.test', 
						help='path to the test data file')
	parser.add_argument('--test-data2', type=str, default='../data/daily547.test', 
						help='path to the test data file')
	parser.add_argument('--trainid-data', type=str, default='../data/userid_oct27.train', 
						help='path to the train data file')
	parser.add_argument('--devid-data', type=str, default='../data/userid_oct27.dev', 
						help='path to the dev data file')
	parser.add_argument('--testid-data', type=str, default='../data/userid_oct27.test', 
						help='path to the test data file')
	parser.add_argument('--testid-data2', type=str, default='../data/userid_daily547.test', 
						help='path to the test data file')
	
	parser.add_argument('--char-dim', type=int, default=50, 
						help='char embedding dimension (default: 30)')
	parser.add_argument('--word-dim', type=int, default=100, 
						help='input dimension (default: 50)')
	parser.add_argument('--hidden-dim', type=int, default=250, 
						help='hidden state dim (default: 100)')
	parser.add_argument('--word-hidden-dim', type=int, default=150, 
						help='hidden state dim (default: 100)')
	parser.add_argument('--char-hidden-dim', type=int, default=100, 
						help='char ltm hidden state dim (default: 50)')
	parser.add_argument('--tag-embed-dim', type=int, default=20, 
						help='tag embedding (default: 20)')
	parser.add_argument('--char-num-of-layers', type=int, default=1, 
						help='char lstm num of layers (default: 1)')
	parser.add_argument('--word-num-of-layers', type=int, default=1, 
						help='number of layers in word lstm (default: 1)')

	parser.add_argument('--log-errors-file', type=str, default='../logs/log_errors.txt', 
						help='path to the log error file')
	parser.add_argument('--batch-size', type=int, default=64, 
						help='input batch size for training (default: 64)')
	parser.add_argument('--vocab-size', type=int, default=4150, 
						help='vocab size (default: 5000)')
	parser.add_argument('--epochs', type=int, default=100, 
						help='number of epochs to train (default: 250)')
	parser.add_argument('--dropout', type=float, default=0.35, 
						help='dropout rate (default: 0.7)')
	parser.add_argument('--lr', type=float, default=0.001, 
						help='learning rate (default: 0.01)')
	parser.add_argument('--optimizer', type=str, default='adam', choices = ['adam','sgd','adadelta','adagrad'],
						help='optimizer: default adam')
	parser.add_argument('--cuda', action='store_true', 
						default=False, help='enables CUDA training')
	parser.add_argument('--write-errors', action='store_true', 
						default=True, help='write errors into a file')

	# training additional features
	parser.add_argument('--use-brown-feats', action='store_true', 
						default=True, help='uses additional brown features during training')
	parser.add_argument('--use-tag-features', action='store_true', 
						default=True, help='uses tagdict features during training')
	parser.add_argument('--use-ud-feats', action='store_true', 
						default=True, help='uses tagdict features during training')
	parser.add_argument('--use-tag-feats', action='store_true', 
						default=False, help='uses tagdict features during training')
	parser.add_argument('--use-name-features', action='store_true', 
						default=False, help='uses name features during training')

	parser.add_argument('--slow', action='store_true', 
						default=True, help='slow running: gets loss stats as well')

	# training overfitting/regularization parameters
	parser.add_argument('--use-regularization', action='store_true', 
						default=True, help='uses l2 regularization during training')
	parser.add_argument('--use-l1', action='store_true', 
						default=False, help='uses l1 regularization during training')
	#parser.add_argument('--l1-reg-factor', type=float, default=1000000, 
	#					help='l1 reg factor (default: 1000000)')
	parser.add_argument('--l2-reg-factor', type=float, default=200, 
						help='l2 reg factor (default: 1000)')

	# additinal preprocessing dataset
	parser.add_argument('--dont-preprocess-data', action='store_true', 
						default=False, help='dont preprocess data')
	parser.add_argument('--dont-lowercase-words', action='store_true', 
						default=False, help='dont lowercase all words')
	parser.add_argument('--combine-train-dev', action='store_true', 
						default=True, help='combine train and dev data for training')
	parser.add_argument('--combine-train-dev-test', action='store_true', 
						default=False, help='combine train, dev and test data for training')



	# additional model parameters
	parser.add_argument('--use-relu', action='store_true', 
						default=False, help='uses relu non-linearity everywhere for FCs')
	parser.add_argument('--use-logistic', action='store_true', 
						default=False, help='uses logistic instead of sigmoid')
	parser.add_argument('--use-pretrained-embed', action='store_true', 
						default=True, help='use pretrained word embedding intact in addition to learning')
	parser.add_argument('--learn-word-embed', action='store_true', 
						default=True, help='learn word embedding')
	parser.add_argument('--share-prev-tag', action='store_true', 
						default=False, help='use previous tag in predicting current tag')
	parser.add_argument('--share-hiddenstates', action='store_true', 
						default=False, help='use previous hidden state in predicting current tag')
	parser.add_argument('--just-pad-sents', action='store_true', 
						default=False, help='use previous hidden state in predicting current tag')
	parser.add_argument('--plots', action='store_true', 
						default=False, help='use bilstm-crf for pos tagging')
	parser.add_argument('--word-embeds', type=str, default='glove', choices = [None,'word2vec','glove','polyglot'],
						help='word embeddings to be used')
	parser.add_argument('--word-embeds-file', type=str, default='data/glove.twitter.27B/glove.twitter.27B.100d.txt',
						help='word embeddings to be used')
	parser.add_argument('--word-embed-dim', type=int, default=100,
						help='word embeddings sizeto be used')

	# model choices, saving and loading, logging
	parser.add_argument('--use-GRU', action='store_true', 
						default=False, help='uses GRU instead of LSTMs')
	parser.add_argument('--use-CRF', action='store_true', 
						default=False, help='uses CRF for viterbi decoding, but still the log loss')
	parser.add_argument('--model', type=str, default='char_bilstm', choices = ['bilstm','char_bilstm','bilstm-crf','char_bilstm-crf'],
						help='model is a bi-lstm on the words')
	parser.add_argument('--load-prev-model', type=str, default=None,
						help='load previous model: filepath')
	parser.add_argument('--save-model', type=str, default=None,
						help='save model: filepath')
	parser.add_argument('--log-file', type=str, default=None,
						help='log file: filepath')

	parser.add_argument('--dynet-mem',type=int, default=1024, 
						help='dynet-mem (default: 1024)')
	parser.add_argument('--dynet-gpus',type=int, default=1, 
						help='dynet-gpus (default: 1)')
	parser.add_argument('--dynet-devices',type=str, default='--dynet-devices', 
						help='dynet-devices (default: 0)')
	parser.add_argument('--dynet-gpu', action='store_true', 
						default=False, help='uses GPU instead of cpu')
	parser.add_argument('--dynet-profiling',type=int, default=0, 
						help='dynet-profiling (default: 0)')
	parser.add_argument('--dynet-autobatch',type=int, default=1, 
						help='--dynet-autobatch (default: 1)')

	parser.add_argument('--use-vae', action='store_true', 
						default=False, help='use vae for ensemble version')
	parser.add_argument('--x-size', type=int, default=50,
						help='x-size')
	parser.add_argument('--h-size', type=int, default=50,
						help='h-size')
	parser.add_argument('--z-size', type=int, default=50,
						help='z-size')
	parser.add_argument('--vae-loss-factor', type=float, default=1, 
						help='vae basis factor (default: 1)')
	parser.add_argument('--use-loss-factor', action='store_true', 
						default=False, help='uses ensemble loss factor')


	parser.add_argument('--node-dim', type=int, default=50, 
						help='node dim (default: 20)')
	parser.add_argument('--early-epochs', type=int, default=15 , 
						help='number of early stopping epochs (default: 15)')
	parser.add_argument('--pretrain-epochs', type=int, default=0, 
						help='pretrain no of epochs rate (default: 3)')
	parser.add_argument('--num-basis', type=int, default=3, 
						help='pretrain no of epochs rate (default: 3)')
	parser.add_argument('--no-ensemble', action='store_true', 
						default=False, help='disables CUDA training')
	parser.add_argument('--just-ensemble', action='store_true', 
						default=False, help='disables CUDA training')

	parser.add_argument('--use-hinge-loss', action='store_true', 
						default=False, help='uses hinge loss during training')
	parser.add_argument('--use-l2-basis', action='store_true', 
						default=True, help='uses l2 regularization for basis matrix during training')
	#parser.add_argument('--use-l1-basis', action='store_true', 
	#					default=False, help='uses l1 regularization for basis matrix during training')
	parser.add_argument('--use-l2-non-auth-vec', action='store_true', 
						default=True, help='uses l2 regularization for non auth vec during training')
	#parser.add_argument('--use-l1-non-auth-vec', action='store_true', 
	#					default=False, help='uses l1 regularization for non auth vec during training')
	parser.add_argument('--use-author-dropout', action='store_true', 
						default=True, help='uses dropout for author vec during training')
	parser.add_argument('--use-node-feature', action='store_true', 
						default=False, help='uses social node author as feature during training')
	#parser.add_argument('--use-l1-mean-loss', action='store_true', 
	#					default=False, help='use l1 mean loss of parameters during training')
	#parser.add_argument('--l1-reg-loss-factor', type=float, default=1000, 
	#					help='l2 reg factor (default: 1000)')
	#parser.add_argument('--dont-use-author-vec', action='store_true', 
	#					default=False, help="don't use author vec")
	parser.add_argument('--only-pretrain', action='store_true', 
						default=False, help="only pretraining")
	parser.add_argument('--no-bias', action='store_true', 
						default=False, help="no bias")
	parser.add_argument('--freeze-char', action='store_true', 
						default=False, help="freezing char embeddings for ensemble")
	parser.add_argument('--freeze-word', action='store_true', 
						default=False, help="freezing word embeddings for ensemble")
	parser.add_argument('--freeze-char-lstm', action='store_true', 
						default=False, help="freezes char level bilstm")
	parser.add_argument('--freeze-word-lstm', action='store_true', 
						default=False, help="freezes word level bilstm")
	parser.add_argument('--freeze-surface-features', action='store_true', 
						default=False, help="freezes surface level features")
	parser.add_argument('--dont-use-author_vec', action='store_true', 
						default=False, help="freezes surface level features")


	parser.add_argument('--l2-basis-factor', type=float, default=200, 
						help='l2 basis factor (default: 2)')
	#parser.add_argument('--l1-basis-factor', type=float, default=1, 
	#					help='l1 basis factor (default: 1)')
	parser.add_argument('--l2-non-auth-vec-factor', type=float, default=200, 
						help='l2 non auth vec factor factor (default: 10000)')
	#parser.add_argument('--l1-non-auth-vec-factor', type=float, default=1000000, 
	#					help='l1 non auth vec factor (default: 1000000)')
	parser.add_argument('--pretrain-non-auth-factor2', type=float, default=1, 
						help='pretrain non auth factor 2 (default: 1)')
	#parser.add_argument('--pretrain-non-auth-factor1', type=float, default=1, 
	#					help='pretrain non auth factor 1 (default: 1)')

	parser.add_argument('--follow-vecs', type=str, default='../data/author_vecs/follow/line/follow_50_combined_pure.emb', 
						help='path to the follow vectors data file')
	parser.add_argument('--mention-vecs', type=str, default='../data/author_vecs/mention/line/mention_50_combined_pure.emb', 
						help='path to the mention vectors data file')
	parser.add_argument('--retweet-vecs', type=str, default='../data/author_vecs/retweet/line/retweet_50_combined_pure.emb', 
						help='path to the retweet vectors data file')
	parser.add_argument('--network', type=str, default='follow', choices = ['follow','mention','retweet'],
						help='network to use: default mention')
	parser.add_argument('--use-all-networks', action='store_true', 
						default=False, help='use all social networks')
	

	args = parser.parse_args()
	return args








