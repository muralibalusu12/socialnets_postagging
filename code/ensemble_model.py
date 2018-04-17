import dynet as dy
import argparse
import logging
import os
import random
from collections import Counter, defaultdict
#import matplotlib.pyplot as plt
import numpy as np
import time




class ensemble_char_BiLSTMNetwork():
	def __init__(self, args, char_to_ix, word_to_ix, tag_to_ix, num_basis, word_vecs = None, char_vecs = None):
		self.args = args
		self.model = dy.Model()
		self.word_vecs = word_vecs
		self.char_vecs = char_vecs
		self.word_to_ix = word_to_ix

		if '<-s->' not in char_to_ix:
			char_to_ix['<-s->'] = len(char_to_ix)
		if '<-e->' not in char_to_ix:
			char_to_ix['<-e->'] = len(char_to_ix)

		'''
		i used fuzzy wuzzy to get the closest words in the brown clusters to all words that is lower cased
		so even if I get new word: I obtain the fuzzy close words and its respective cluster from the brown clusters
		use prefixes as features: quite a long vector, but helps a lot
		'''

		if self.args.use_brown_feats:
			self.brown_prefix={}
			f = open('utils/brown_prefix.txt','r')
			for line in f:
				self.brown_prefix[line.strip()]=len(self.brown_prefix)
			f.close()
			self.brown_feats={}
			f = open('utils/final_brown_feats.txt','r')
			for line in f:
				word = line.strip().split('\t')[0]
				prefix=line.strip().split('\t')[1]
				self.brown_feats[word]=prefix 
			f.close()

			self.close_brown_words={}
			file = 'utils/brown_feats.txt'
			f = open(file,'r')
			for line in f:
				content = line.strip().split('\t')
				word1=content[0]
				word2=content[1]
				self.close_brown_words[word1]=word2

		if self.args.use_tag_features:
			self.close_brown_words={}
			file = 'utils/brown_feats.txt'
			f = open(file,'r')
			for line in f:
				content = line.strip().split('\t')
				word1=content[0]
				word2=content[1]
				self.close_brown_words[word1]=word2

			file1 = 'utils/all_tags.txt'
			file2 = 'utils/word_tag_data.txt'

			file3 = 'utils/all_tags_ud.txt'
			file4 = 'utils/word_tag_data_ud.txt'

			self.feat_tags1={}
			f = open(file1, 'r')
			for line in f:
				self.feat_tags1[line.strip()]=len(self.feat_tags1)
			f.close()

			self.word_tags1={}
			f=open(file2, 'r')
			for line in f:
				content = line.strip().split()
				if len(content)>0:
					word = content[0]
					for t in content[1:]:
						if word in self.word_tags1:
							self.word_tags1[word].append(t)
						else:
							self.word_tags1[word]=[]
							self.word_tags1[word].append(t)
			f.close()

			if self.args.use_ud_feats:

				self.feat_tags2={}
				f = open(file3, 'r')
				for line in f:
					self.feat_tags2[line.strip()]=len(self.feat_tags2)
				f.close()

				self.word_tags2={}
				f=open(file4, 'r')
				for line in f:
					content = line.strip().split()
					if len(content)>0:
						word = content[0]
						for t in content[1:]:
							if word in self.word_tags2:
								self.word_tags2[word].append(t)
							else:
								self.word_tags2[word]=[]
								self.word_tags2[word].append(t)
				f.close()

				self.tag_feats_length = len(self.feat_tags1)+len(self.feat_tags2)

			else:
				self.tag_feats_length = len(self.feat_tags1)
		
		logging.info('finalized vocab size: '+str(args.vocab_size))

		self.tag_to_ix = tag_to_ix
		self.char_to_ix = char_to_ix

		#self.char_to_ix['<pad>'] = len(self.char_to_ix)
		self.ix_to_tag = {}
		for k,v in self.tag_to_ix.items():
			self.ix_to_tag[v] = k

		self.tagset_size = len(self.tag_to_ix)

		if self.args.no_ensemble:
			self.num_basis=1
		else:
			self.num_basis=num_basis

		self.char_embeddings=[]
		self.embeddings=[]
		self.tag_embeddings=[]
		self.char_bi_lstm = []
		self.char_output_w=[]
		self.char_output_b=[]
		self.char_output_w2=[]
		self.char_output_b2=[]
		self.word_bi_lstm = []
		self.output_w0=[]
		self.output_b0=[]
		self.output_w=[]
		self.output_b=[]
		self.output_w2=[]
		self.output_b2=[]
		self.output_w3=None
		self.output_b3=None
		self.output_w4=None
		self.output_b4=None

		for i in range(self.num_basis):
			
			##################
			# learning character embeddings
			##################
			self.char_embeddings.append(self.model.add_lookup_parameters((len(self.char_to_ix),self.args.char_dim)))
			
			##################
			# learning word embeddings #
			##################
			if self.args.learn_word_embed:
				self.embeddings.append(self.model.add_lookup_parameters((args.vocab_size, self.args.word_dim)))
			else:
				self.embeddings = None

			if self.args.share_prev_tag:
				self.tag_embeddings.append(self.model.add_lookup_parameters((self.tagset_size+1, args.tag_embed_dim)))
			
			##################
			# char-level-RNN #
			##################
			if args.use_GRU:
				self.char_bi_lstm.append(dy.BiRNNBuilder(args.char_num_of_layers, args.char_dim, args.char_hidden_dim, self.model, dy.GRUBuilder))
			else:
				self.char_bi_lstm.append(dy.BiRNNBuilder(args.char_num_of_layers, args.char_dim, args.char_hidden_dim, self.model, dy.VanillaLSTMBuilder))

			self.char_output_w.append(self.model.add_parameters((args.char_hidden_dim, args.char_hidden_dim)))
			self.char_output_b.append(self.model.add_parameters((args.char_hidden_dim)))

			self.char_output_w2.append(self.model.add_parameters((args.char_hidden_dim, args.char_hidden_dim)))
			self.char_output_b2.append(self.model.add_parameters((args.char_hidden_dim)))

			##################
			# word-level-RNN #
			##################
			if args.use_pretrained_embed and args.learn_word_embed:
				char_output_hidden_dim = args.char_hidden_dim + args.word_dim + self.args.word_embed_dim
			elif args.use_pretrained_embed==False and args.learn_word_embed:
				char_output_hidden_dim = args.char_hidden_dim + args.word_dim
			elif args.use_pretrained_embed and args.learn_word_embed==False:
				char_output_hidden_dim = args.char_hidden_dim + self.args.word_embed_dim
			else:
				char_output_hidden_dim = args.char_hidden_dim

			if args.use_GRU:
				self.word_bi_lstm.append(dy.BiRNNBuilder(args.word_num_of_layers, char_output_hidden_dim, args.word_hidden_dim, self.model, dy.GRUBuilder))
			else:
				self.word_bi_lstm.append(dy.BiRNNBuilder(args.word_num_of_layers, char_output_hidden_dim, args.word_hidden_dim, self.model, dy.VanillaLSTMBuilder))

			##################
			# project the rnn hidden state to a vector of tagset_size length #
			##################
			if self.args.share_prev_tag:
				output_hidden_dim = args.word_hidden_dim + args.tag_embed_dim
			elif self.args.share_hiddenstates:
				output_hidden_dim = args.word_hidden_dim + args.word_hidden_dim + args.word_hidden_dim
			else:
				output_hidden_dim = args.word_hidden_dim

			##using previous, current, next for now: hence x 3
			if not self.args.freeze_surface_features:
				if self.args.use_tag_features:
					output_hidden_dim += self.tag_feats_length*3
				if self.args.use_brown_feats:
					output_hidden_dim += len(self.brown_prefix)*3
				if self.args.use_tag_feats:
					output_hidden_dim += self.tag_feats_length*5
				if self.args.use_name_features:
					output_hidden_dim += self.name_length*5
			else:
				output_hidden_dim2 = 0
				if self.args.use_tag_features:
					output_hidden_dim2 += self.tag_feats_length*3
				if self.args.use_brown_feats:
					output_hidden_dim2 += len(self.brown_prefix)*3
				if self.args.use_tag_feats:
					output_hidden_dim2 += self.tag_feats_length*5
				if self.args.use_name_features:
					output_hidden_dim2 += self.name_length*5

				self.output_w3 = self.model.add_parameters((self.tagset_size*16, output_hidden_dim2))
				self.output_b3 = self.model.add_parameters((self.tagset_size*16))
				self.output_w4 = self.model.add_parameters((self.tagset_size*4, self.tagset_size*16))
				self.output_b4 = self.model.add_parameters((self.tagset_size*4))
				self.output_w5 = self.model.add_parameters((self.tagset_size, self.tagset_size*4))
				self.output_b5 = self.model.add_parameters((self.tagset_size))

			##################
			# final fully connected layers leading to the soft-max #
			##################
			#self.output_w00 = self.model.add_parameters((args.hidden_dim*4, output_hidden_dim))
			#self.output_b00 = self.model.add_parameters((args.hidden_dim*4))
			self.output_w0.append(self.model.add_parameters((args.hidden_dim*2, output_hidden_dim)))
			self.output_b0.append(self.model.add_parameters((args.hidden_dim*2)))
			self.output_w.append(self.model.add_parameters((args.hidden_dim, args.hidden_dim*2)))
			self.output_b.append(self.model.add_parameters((args.hidden_dim)))
			self.output_w2.append(self.model.add_parameters((self.tagset_size, args.hidden_dim)))
			self.output_b2.append(self.model.add_parameters((self.tagset_size)))

		k=self.num_basis

		

		if self.args.use_all_networks:
			self.basis_matrix = self.model.add_parameters((k,self.args.node_dim*3), init=dy.NormalInitializer())
			self.basis_bias = self.model.add_parameters((k), init=dy.NormalInitializer())
			self.non_author_vector = self.model.add_lookup_parameters((3, self.args.node_dim),init=dy.NormalInitializer())
		else:
			self.basis_matrix = self.model.add_parameters((k,self.args.node_dim), init=dy.NormalInitializer())
			self.basis_bias = self.model.add_parameters((k), init=dy.NormalInitializer())
			self.non_author_vector = self.model.add_lookup_parameters((1, self.args.node_dim),init=dy.NormalInitializer())

		

		if self.args.use_all_networks:
			self.random_basis_vectors = self.model.add_lookup_parameters((k, self.args.node_dim*3),init=dy.NormalInitializer())
		else:
			self.random_basis_vectors = self.model.add_lookup_parameters((k, self.args.node_dim),init=dy.NormalInitializer())

		if self.args.use_vae:
			self.vae_encode_layer1 = self.model.add_parameters((self.args.h_size, self.args.x_size))
			self.vae_encode_bias1 = self.model.add_parameters((self.args.h_size))
			#self.vae_encode_layer2 = self.model.add_parameters((self.args.h_size, self.args.h_size))
			#self.vae_encode_bias2 = self.model.add_parameters((self.args.h_size))
			self.vae_encode_layer3 = self.model.add_parameters((self.args.z_size*2, self.args.h_size))
			self.vae_encode_bias3 = self.model.add_parameters((self.args.z_size*2))

			self.vae_decode_layer1 = self.model.add_parameters((self.args.h_size, self.args.z_size))
			self.vae_decode_bias1 = self.model.add_parameters((self.args.h_size))
			self.vae_decode_layer2 = self.model.add_parameters((self.args.h_size, self.args.h_size))
			self.vae_decode_bias2 = self.model.add_parameters((self.args.h_size))
			self.vae_decode_layer3 = self.model.add_parameters((self.args.x_size, self.args.h_size))
			self.vae_decode_bias3 = self.model.add_parameters((self.args.x_size))

			#self.vae_encode2 = self.model.add_parameters((self.tagset_size, len(self.brown_prefix)))

		if self.args.load_prev_model is not None:
			self.model.populate(args.load_prev_model)

	def regularize_weights(self, pretrain=False):

		#prev_time = time.time()
		
		#'''
		cc=0
		for k in range(self.num_basis):
			kk=k

			if self.args.freeze_char:
				k=0
				if cc==0:
					l2reg1 = dy.squared_norm(self.char_embeddings[k].expr())*dy.scalarInput(1/self.num_basis)
					cc=1
				else:
					l2reg1 += dy.squared_norm(self.char_embeddings[k].expr())*dy.scalarInput(1/self.num_basis)
			else:
				if cc==0:
					l2reg1 = dy.squared_norm(self.char_embeddings[k].expr())
					cc=1
				else:
					l2reg1 += dy.squared_norm(self.char_embeddings[k].expr())

			k=kk
			if self.args.freeze_word:
				k=0
				if self.args.learn_word_embed:
					l2reg1 += dy.squared_norm(self.embeddings[k].expr())*dy.scalarInput(1/self.num_basis)
			else:
				if self.args.learn_word_embed:
					l2reg1 += dy.squared_norm(self.embeddings[k].expr())

			k=kk
			if self.args.freeze_char_lstm:
				k=0
				for i in range(self.args.char_num_of_layers):
					for ex2 in self.char_bi_lstm[k].builder_layers[i][0].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)*dy.scalarInput(1/self.num_basis)
					for ex2 in self.char_bi_lstm[k].builder_layers[i][1].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)*dy.scalarInput(1/self.num_basis)
			else:
				for i in range(self.args.char_num_of_layers):
					for ex2 in self.char_bi_lstm[k].builder_layers[i][0].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)
					for ex2 in self.char_bi_lstm[k].builder_layers[i][1].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)

			k=kk
			if self.args.freeze_word_lstm:
				k=0
				for i in range(self.args.word_num_of_layers):
					for ex2 in self.word_bi_lstm[k].builder_layers[i][0].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)*dy.scalarInput(1/self.num_basis)
					for ex2 in self.word_bi_lstm[k].builder_layers[i][1].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)*dy.scalarInput(1/self.num_basis)
			else:
				for i in range(self.args.word_num_of_layers):
					for ex2 in self.word_bi_lstm[k].builder_layers[i][0].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)
					for ex2 in self.word_bi_lstm[k].builder_layers[i][1].get_parameter_expressions():
						for ex in ex2:
							l2reg1 += dy.squared_norm(ex)

			
			k=kk
			l2reg1 += dy.squared_norm(self.char_output_w[k].expr())
			l2reg1 += dy.squared_norm(self.char_output_b[k].expr())
			l2reg1 += dy.squared_norm(self.char_output_w2[k].expr())
			l2reg1 += dy.squared_norm(self.char_output_b2[k].expr())

			
			l2reg1 += dy.squared_norm(self.output_w0[k].expr())*dy.scalarInput(1/self.num_basis)
			l2reg1 += dy.squared_norm(self.output_b0[k].expr())*dy.scalarInput(1/self.num_basis)
			k = 0
			l2reg1 += dy.squared_norm(self.output_w[k].expr())*dy.scalarInput(1/self.num_basis)
			l2reg1 += dy.squared_norm(self.output_b[k].expr())*dy.scalarInput(1/self.num_basis)
			l2reg1 += dy.squared_norm(self.output_w2[k].expr())*dy.scalarInput(1/self.num_basis)
			l2reg1 += dy.squared_norm(self.output_b2[k].expr())*dy.scalarInput(1/self.num_basis)
			k=kk

			if self.args.freeze_surface_features:
				l2reg1 += dy.squared_norm(self.output_w3.expr())*dy.scalarInput(1/self.num_basis)
				l2reg1 += dy.squared_norm(self.output_b3.expr())*dy.scalarInput(1/self.num_basis)
				l2reg1 += dy.squared_norm(self.output_w4.expr())*dy.scalarInput(1/self.num_basis)
				l2reg1 += dy.squared_norm(self.output_b4.expr())*dy.scalarInput(1/self.num_basis)
				l2reg1 += dy.squared_norm(self.output_w5.expr())*dy.scalarInput(1/self.num_basis)
				l2reg1 += dy.squared_norm(self.output_b5.expr())*dy.scalarInput(1/self.num_basis)

			if k==0:
				reg_loss1 =l2reg1*dy.scalarInput(1/self.args.l2_reg_factor)
			else:
				reg_loss1 +=l2reg1*dy.scalarInput(1/self.args.l2_reg_factor)

		if self.args.use_vae:
			l2regv = dy.squared_norm(self.vae_encode_layer1.expr())
			l2regv += dy.squared_norm(self.vae_encode_bias1.expr())
			#l2regv += dy.squared_norm(self.vae_encode_layer2.expr())
			#l2regv += dy.squared_norm(self.vae_encode_bias2.expr())
			l2regv += dy.squared_norm(self.vae_encode_layer3.expr())
			l2regv += dy.squared_norm(self.vae_encode_bias3.expr())

			l2regv += dy.squared_norm(self.vae_decode_layer1.expr())
			l2regv += dy.squared_norm(self.vae_decode_bias1.expr())
			l2regv += dy.squared_norm(self.vae_decode_layer2.expr())
			l2regv += dy.squared_norm(self.vae_decode_bias2.expr())
			l2regv += dy.squared_norm(self.vae_decode_layer3.expr())
			l2regv += dy.squared_norm(self.vae_decode_bias3.expr())

			reg_loss1 += l2regv*dy.scalarInput(1/self.args.l2_reg_factor)
		
		if self.args.use_l2_basis:
			l2reg = dy.squared_norm(self.basis_matrix.expr())
			l2reg += dy.squared_norm(self.basis_bias.expr())

			reg_loss2 =l2reg*dy.scalarInput(1/self.args.l2_basis_factor)
		
		if self.args.use_l2_non_auth_vec:
			l2reg2 = dy.squared_norm(self.non_author_vector.expr())

			reg_loss4 =l2reg2*dy.scalarInput(1/self.args.l2_non_auth_vec_factor)

		if pretrain:
			if not self.args.dont_use_author_vec:
				if self.args.use_l2_non_auth_vec:
					reg_loss1+=reg_loss4*dy.scalarInput(1/self.args.pretrain_non_auth_factor2)
				#if self.args.use_l1_non_auth_vec:
				#	reg_loss1+=reg_loss5/self.args.pretrain_non_auth_factor1
			#curr_time = time.time()
			#print ('time for one reg training: ', curr_time - prev_time)

			#if self.args.use_l1_mean_loss:
			#	return reg_loss1 + self.l1_reg_loss()
			return reg_loss1
		else:
			if self.args.use_l2_basis:
				reg_loss1+=reg_loss2

			#if self.args.use_l1_basis:
			#	reg_loss1+=reg_loss3

			if not self.args.dont_use_author_vec:
				if self.args.use_l2_non_auth_vec:
					reg_loss1+=reg_loss4

				#if self.args.use_l1_non_auth_vec:
				#	reg_loss1+=reg_loss5

			#if self.args.use_l1_mean_loss:
			#	return reg_loss1 + self.l1_reg_loss()

			return reg_loss1

	def l1_reg_loss(self):
		pass
		

	# preprocessing function for all inputs
	def _preprocess_input(self, sentence, to_ix):
		if 'unk' in to_ix: # for words in tweet
			return [to_ix[word] if word in to_ix else to_ix['unk'] for word in sentence]
		else: # for tags of the words
			return [to_ix[word] for word in sentence]

	def _construct_feature1(self,sentence):
		feature_vector=[]
		length = len(self.brown_prefix)
		feats_vec=[]
		feat_vec=[0]*length
		feats_vec.append(feat_vec)
		feats_vec.append(feat_vec)
		for i,word in enumerate(sentence):
			feat_vec=[0]*length
			if word in self.brown_feats:
				cluster = self.brown_feats[word]
				i=2
				while i<=len(cluster):
					feat_vec[self.brown_prefix[cluster[0:i]]]=1
					i+=2
				feat_vec[self.brown_prefix[cluster]]=1
			else:
				if word in self.close_brown_words:
					word = self.close_brown_words[word]
					if word in self.brown_feats:
						cluster = self.brown_feats[word]
						i=2
						while i<=len(cluster):
							feat_vec[self.brown_prefix[cluster[0:i]]]=1
							i+=2
						feat_vec[self.brown_prefix[cluster]]=1
			feats_vec.append(feat_vec)
			#feature_vector.append(dy.inputTensor(feat_vec))
		feat_vec=[0]*length
		feats_vec.append(feat_vec)
		feats_vec.append(feat_vec)
		#return feature_vector
		feature_vector = []
		for f0, f1, f2, f3, f4 in zip(feats_vec[:-4],feats_vec[1:-3],feats_vec[2:-2], feats_vec[3:-1],feats_vec[4:]):
			feature_vector.append(dy.inputTensor(f1+f2+f3))



			#else:
				# use close_brown_words
			#feature_vector.append(dy.inputTensor(feat_vec))

		return feature_vector

	def _construct_feature2(self,sentence):
		feature_vector=[]
		if self.args.use_ud_feats:
			length = len(self.feat_tags1)+len(self.feat_tags2)
		else:
			length = len(self.feat_tags1)
		midlen = len(self.feat_tags1)

		feat_vec=[0]*length
		feat_vecs=[]
		feat_vecs.append(feat_vec)
		feat_vecs.append(feat_vec)

		for word in sentence:
			feat_vec=[0]*length

			if word in self.word_tags1:
				maxt = len(self.word_tags1[word])
				for i,t in enumerate(self.word_tags1[word]):
					feat_vec[self.feat_tags1[t]]=((maxt-i)*1.0)/maxt
			else:
				if word in self.close_brown_words:
					close_word = self.close_brown_words[word]
					if close_word in self.word_tags1:
						maxt = len(self.word_tags1[close_word])
						for i,t in enumerate(self.word_tags1[close_word]):
							feat_vec[self.feat_tags1[t]]=((maxt-i)*1.0)/maxt

			if self.args.use_ud_feats:
				if word in self.word_tags2:
					maxt = len(self.word_tags2[word])
					for i,t in enumerate(self.word_tags2[word]):
						feat_vec[midlen+self.feat_tags2[t]]=((maxt-i)*1.0)/maxt
				else:
					if word in self.close_brown_words:
						close_word = self.close_brown_words[word]
						if close_word in self.word_tags2:
							maxt = len(self.word_tags2[close_word])
							for i,t in enumerate(self.word_tags2[close_word]):
								feat_vec[midlen+self.feat_tags2[t]]=((maxt-i)*1.0)/maxt

			feat_vecs.append(feat_vec)

		feat_vec=[0]*length
		feat_vecs.append(feat_vec)
		feat_vecs.append(feat_vec)

		for f0,f1,f2,f3,f4 in zip(feat_vecs[:-4],feat_vecs[1:-3],feat_vecs[2:-2],feat_vecs[3:-1],feat_vecs[4:]):
			feature_vector.append(dy.inputTensor(f1+f2+f3))

		#feature_vector.append(dy.inputTensor(feat_vec))

		return feature_vector
	
	def _construct_feature(self, sentence):
		feature_vector1 = self._construct_feature1(sentence)
		feature_vector2 = self._construct_feature2(sentence)

		feature_vector=[]
		for i,j in zip(feature_vector1,feature_vector2):
			feature_vector.append(dy.concatenate([i,j]))

		return feature_vector

	# embed the sentence with embeddings(look up parameters) for each word
	def _embed_sentence(self, sentence, k):
		#return [self.embeddings[word] for word in sentence]
		if self.args.freeze_word:
			k=0
		return [dy.lookup(self.embeddings[k], word) for word in sentence]
	
	# embed the sentence with embeddings(look up parameters) for each word
	def _embed_word(self, sentence, k):
		#return [self.embeddings[word] for word in sentence]
		if self.args.freeze_char:
			k=0
		return [dy.lookup(self.char_embeddings[k], word) for word in sentence]
	
	# embed the sentence with embeddings(look up parameters) for each word
	def _embed_output(self, sentence):
		embeds = [dy.lookup(self.tag_embeddings, word) for word in sentence][:-1]
		start_embed = [dy.lookup(self.tag_embeddings, len(self.tag_to_ix)) ]
		return start_embed + embeds

	# run rnn on the specified input with the corresponding init state
	def _run_rnn(self, init_state, input_vecs):
		s = init_state
		states = s.transduce(input_vecs)
		rnn_outputs = [s for s in states]
		return rnn_outputs

	# run rnn on the specified input with the corresponding init state
	def _run_rnn2(self, init_state, input_vecs):
		s = init_state
		states = s.add_inputs(input_vecs)
		rnn_outputs = [(s1.h()[0],s2.h()[0]) for s1,s2 in states]
		return rnn_outputs

	# get probabilities of the tags in the tagset by using an mlp over the hidden state
	def _get_probs(self, rnn_output, k):
		
		output_w0 = dy.parameter(self.output_w0[k])
		output_b0 = dy.parameter(self.output_b0[k])
		k = 0
		output_w = dy.parameter(self.output_w[k])
		output_b = dy.parameter(self.output_b[k])
		output_w2 = dy.parameter(self.output_w2[k])
		output_b2 = dy.parameter(self.output_b2[k])

		#temp_output = dy.tanh(output_w00 * rnn_output + output_b00)
		if self.args.use_relu:
			temp_output = dy.rectify(output_w0 * rnn_output + output_b0)
			temp_output = dy.rectify(output_w * temp_output + output_b)
			temp_output = output_w2 * temp_output + output_b2
		else:
			temp_output = dy.tanh(output_w0 * rnn_output + output_b0)
			temp_output = dy.tanh(output_w * temp_output + output_b)
			temp_output = output_w2 * temp_output + output_b2
		probs = dy.softmax(temp_output)
		return probs, temp_output

	# get probabilities of the tags in the tagset by using an mlp over the hidden state
	def _get_surface_probs(self, rnn_output, k, surface_feature_vector):
		
		output_w0 = dy.parameter(self.output_w0[k])
		output_b0 = dy.parameter(self.output_b0[k])
		k = 0
		output_w = dy.parameter(self.output_w[k])
		output_b = dy.parameter(self.output_b[k])
		output_w2 = dy.parameter(self.output_w2[k])
		output_b2 = dy.parameter(self.output_b2[k])

		if self.args.use_relu:
			temp_output = dy.rectify(output_w0 * rnn_output + output_b0)
			temp_output = dy.rectify(output_w * temp_output + output_b)
			temp_output = output_w2 * temp_output + output_b2
		else:
			temp_output = dy.tanh(output_w0 * rnn_output + output_b0)
			temp_output = dy.tanh(output_w * temp_output + output_b)
			temp_output = output_w2 * temp_output + output_b2

		prob = dy.softmax(temp_output)
		#prob = dy.exp(prob)


		output_w3 = dy.parameter(self.output_w3)
		output_b3 = dy.parameter(self.output_b3)
		output_w4 = dy.parameter(self.output_w4)
		output_b4 = dy.parameter(self.output_b4)
		output_w5 = dy.parameter(self.output_w5)
		output_b5 = dy.parameter(self.output_b5)

		temp_output2 = dy.tanh(output_w3 * surface_feature_vector + output_b3)
		temp_output2 = dy.tanh(output_w4 * temp_output2 + output_b4)
		temp_output2 = output_w5 * temp_output2 + output_b5
		surface_prob = dy.softmax(temp_output2)

		final_prob = surface_prob + prob
		score_prob = temp_output + temp_output2

		return final_prob, score_prob

		#surface_output = ()/(self.num_basis)
		#log_prob = log_prob + dy.softmax(surface_output)
		#final_prob = prob + dy.exp(surface_output)
		#return log_prob, final_prob

	# predicting the next max tag: greedy decoding here
	def _predict(self, probs):
		probs = probs.value()
		idx = probs.index(max(probs))
		return self.ix_to_tag[idx]

	def encoder(self, x):
		vae_encode_layer1 = dy.parameter(self.vae_encode_layer1)
		vae_encode_bias1 = dy.parameter(self.vae_encode_bias1)
		#vae_encode_layer2 = dy.parameter(self.vae_encode_layer2)
		#vae_encode_bias2 = dy.parameter(self.vae_encode_bias2)
		vae_encode_layer3 = dy.parameter(self.vae_encode_layer3)
		vae_encode_bias3 = dy.parameter(self.vae_encode_bias3)

		temp_output = dy.tanh(vae_encode_layer1 * x + vae_encode_bias1)
		#temp_output = dy.tanh(vae_encode_layer2 * temp_output + vae_encode_bias2)
		temp_output = vae_encode_layer3 * temp_output + vae_encode_bias3

		z_mu = temp_output[0:self.args.z_size]
		z_logvar = temp_output[self.args.z_size:]

		return z_mu, z_logvar

	def sample(self, z_mu, z_logvar):
		z_stdev = dy.sqrt(dy.exp(z_logvar))
		z_random = dy.random_normal(self.args.z_size)

		return dy.cmult(z_stdev,z_random) + z_mu

	def decode(self, z):
		vae_decode_layer1 = dy.parameter(self.vae_decode_layer1)
		vae_decode_bias1 = dy.parameter(self.vae_decode_bias1)
		vae_decode_layer2 = dy.parameter(self.vae_decode_layer2)
		vae_decode_bias2 = dy.parameter(self.vae_decode_bias2)
		vae_decode_layer3 = dy.parameter(self.vae_decode_layer3)
		vae_decode_bias3 = dy.parameter(self.vae_decode_bias3)

		temp_output = dy.tanh(vae_decode_layer1 * z + vae_decode_bias1)
		temp_output = dy.tanh(vae_decode_layer2 * temp_output + vae_decode_bias2)
		temp_output = vae_decode_layer3 * temp_output + vae_decode_bias3

		x_recon = temp_output

		return x_recon

	def vae_forward(self, x):
		z_mu, z_logvar = self.encoder(x)
		z = self.sample(z_mu, z_logvar)
		x_recon = self.decode(z)

		return x_recon, z_mu, z_logvar, z

	def vae_loss(self, x, x_recon, z_mu, z_logvar, kld_factor=0.1):
		vec = dy.ones(self.args.z_size)#[1 for i in range(self.args.z_size)]
		l2_loss = dy.squared_distance(x, x_recon)
		kld_element = dy.scalarInput(-0.5)*dy.sum_elems((dy.square(z_mu) + dy.exp(z_logvar))*dy.scalarInput(-1) + vec + z_logvar)
		#z = self.sample(z_mu, z_logvar)

		return l2_loss + kld_element*dy.scalarInput(kld_factor), l2_loss

	def get_full_loss(self, input_sentence, original_input, output_sentence, author_vec, pretrain, return_val=True):
		if self.args.use_all_networks:
			if author_vec[0]:
				author_vec1 = dy.inputTensor(author_vec[0])
			else:
				author_vec1 = dy.lookup(self.non_author_vector, 0, update=True)

			if author_vec[1]:
				author_vec2 = dy.inputTensor(author_vec[1])
			else:
				author_vec2 = dy.lookup(self.non_author_vector, 1, update=True)

			if author_vec[2]:
				author_vec3 = dy.inputTensor(author_vec[2])
			else:
				author_vec3 = dy.lookup(self.non_author_vector, 2, update=True)

			author_vec = dy.concatenate([author_vec1,author_vec2,author_vec3])

		else:
			if author_vec:
				author_vec = dy.inputTensor(author_vec)
				
			else:
				author_vec = dy.lookup(self.non_author_vector, 0, update=True)

		x_recon, z_mu, z_logvar, z = self.vae_forward(author_vec)
		vae_loss, l2_loss = self.vae_loss(author_vec, x_recon, z_mu, z_logvar, kld_factor=0.1)

		loss,aa = self.get_loss(input_sentence, original_input, output_sentence, z, pretrain, return_val)

		if self.args.use_loss_factor:
			loss = loss*dy.scalarInput(1/len(output_sentence))

		return dy.scalarInput(self.args.vae_loss_factor)*vae_loss+loss, aa, dy.scalarInput(self.args.vae_loss_factor)*vae_loss, dy.scalarInput(self.args.vae_loss_factor)*l2_loss


	# run the forward and backward lstm, then obtains probs over tags, calculates the loss and returns it
	def get_loss(self, input_sentence, original_input, output_sentence, author_vec, pretrain, return_val=False):

		#prev_time = time.time()

		if self.args.use_brown_feats and self.args.use_tag_features:
			surface_feature_vector = self._construct_feature(input_sentence)
		elif self.args.use_brown_feats:
			brown_feature_vector = self._construct_feature1(input_sentence)
		elif self.args.use_tag_features:
			tag_feature_vector = self._construct_feature2(input_sentence)

		author_vec_check = 1

		if not self.args.use_vae:
			if self.args.use_all_networks:
				if author_vec[0]:
					author_vec1 = dy.inputTensor(author_vec[0])
				else:
					author_vec1 = dy.lookup(self.non_author_vector, 0, update=True)

				if author_vec[1]:
					author_vec2 = dy.inputTensor(author_vec[1])
				else:
					author_vec2 = dy.lookup(self.non_author_vector, 1, update=True)

				if author_vec[2]:
					author_vec3 = dy.inputTensor(author_vec[2])
				else:
					author_vec3 = dy.lookup(self.non_author_vector, 2, update=True)

				author_vec = dy.concatenate([author_vec1,author_vec2,author_vec3])

			else:
				if author_vec:
					author_vec = dy.inputTensor(author_vec)
					if self.args.dont_use_author_vec:
						author_vec_check = 1
					if self.args.just_ensemble:
						author_vec_check = None
				else:
					author_vec = dy.lookup(self.non_author_vector, 0, update=True)
					if self.args.dont_use_author_vec or self.args.just_ensemble:
						author_vec_check = None

		if self.args.use_author_dropout:
			author_vec = dy.dropout(author_vec, self.args.dropout)

		if self.args.no_ensemble or self.args.just_ensemble:
			pass
		else:
			if pretrain:
				rand_basis_vecs = [dy.lookup(self.random_basis_vectors, k, update=False) for k in range(self.num_basis)]
				if self.args.only_pretrain:
					ensemble_weights = dy.logistic(dy.concatenate([dy.dot_product(author_vec,vec) for vec in rand_basis_vecs]))
					aa = ensemble_weights.value()
				else:
					ensemble_weights = [dy.logistic(dy.dot_product(author_vec,vec)) for vec in rand_basis_vecs]
					aa = [ew.value() for ew in ensemble_weights]
				
			else:
				basis_mx = dy.parameter(self.basis_matrix)
				basis_bs = dy.parameter(self.basis_bias)
				if self.args.no_bias:
					if self.args.use_logistic:
						ensemble_weights = dy.logistic(basis_mx*author_vec)
					else:
						ensemble_weights = dy.softmax(basis_mx*author_vec)
				else:
					if self.args.use_logistic:
						ensemble_weights = dy.logistic(basis_mx*author_vec + basis_bs)
					else:
						ensemble_weights = dy.softmax(basis_mx*author_vec+ basis_bs)

				aa = ensemble_weights.value()

		if self.args.dont_use_author_vec or self.args.just_ensemble:
			v=[]
			for ii in range(self.num_basis):
				v.append(1/self.num_basis)

			if not author_vec_check:
				ensemble_weights = dy.inputTensor(v)
				aa = ensemble_weights.value()

		input_input = input_sentence
		input_sentence = self._preprocess_input(input_sentence, self.word_to_ix)
		output_sentence = self._preprocess_input(output_sentence, self.tag_to_ix)

		for i in range(self.num_basis):
			self.char_bi_lstm[i].set_dropout(self.args.dropout)
			self.word_bi_lstm[i].set_dropout(self.args.dropout)

		all_probs=[]
		rnn_probs=[]
		for k in range(self.num_basis):
			kk=k
			
			word_embeds=[]
			for i in range(len(original_input)):
				word = original_input[i]
				chars = ['<-s->'] + list(word) + ['<-e->']
				input_chars = self._preprocess_input(chars, self.char_to_ix)
				embed_chars = self._embed_word(input_chars,k)
				embed_chars = [dy.dropout(embed, self.args.dropout) for embed in embed_chars]
				k=kk
				if self.args.freeze_char_lstm:
					k=0
				char_rnn_states = self._run_rnn2(self.char_bi_lstm[k], embed_chars)
				word_embed = dy.concatenate([char_rnn_states[1][1], char_rnn_states[-2][0]])

				char_output_w = dy.parameter(self.char_output_w[k])
				char_output_b = dy.parameter(self.char_output_b[k])
				char_output_w2 = dy.parameter(self.char_output_w2[k])
				char_output_b2 = dy.parameter(self.char_output_b2[k])
				word_embed = dy.tanh(char_output_w * word_embed + char_output_b)
				word_embed = char_output_w2 * word_embed + char_output_b2
				word_embeds.append(word_embed)

			if self.args.use_pretrained_embed and self.args.learn_word_embed:
				embedded_sentence = self._embed_sentence(input_sentence,k)
				input_embed = [dy.inputTensor(self.word_vecs[word]) if word in self.word_vecs else dy.inputTensor(self.word_vecs['unk']) for word in input_input ]
				embeds = [dy.concatenate([i,j,k]) for i,j,k in zip(word_embeds,embedded_sentence, input_embed)]

			elif self.args.use_pretrained_embed==False and self.args.learn_word_embed:
				embedded_sentence = self._embed_sentence(input_sentence,k)
				embeds = [dy.concatenate([i,j]) for i,j in zip(word_embeds,embedded_sentence)]

			elif self.args.use_pretrained_embed and self.args.learn_word_embed==False:
				input_embed = [dy.inputTensor(self.word_vecs[word]) if word in self.word_vecs else dy.inputTensor(self.word_vecs['unk']) for word in input_input ]
				embeds = [dy.concatenate([i,k]) for i,k in zip(word_embeds, input_embed)]

			else:
				embeds = word_embeds

			embeds = [dy.dropout(embed, self.args.dropout) for embed in embeds]

			k=kk
			if self.args.freeze_word_lstm:
				k=0
			rnn_outputs = self._run_rnn(self.word_bi_lstm[k], embeds)


			p=[]
			rnn_p=[]
			i=0
			for rnn_output, output_char in zip(rnn_outputs, output_sentence):

				if self.args.use_node_feature:
					rnn_output = dy.concatenate([rnn_output, author_vec])

				rnn_output = dy.dropout(rnn_output, self.args.dropout)
				if self.args.freeze_surface_features:
					if self.args.use_tag_features and self.args.use_brown_feats:
						probs, rnn_prob = self._get_surface_probs(rnn_output, k, surface_feature_vector[i])
					elif self.args.use_brown_feats:
						probs, rnn_prob = self._get_surface_probs(rnn_output, k, brown_feature_vector[i])
					elif self.args.use_tag_features:
						probs, rnn_prob = self._get_surface_probs(rnn_output, k, tag_feature_vector[i])
					i+=1
				else:	
					if self.args.use_tag_features and self.args.use_brown_feats:
						rnn_output = dy.concatenate([rnn_output, surface_feature_vector[i]])
					elif self.args.use_brown_feats:
						rnn_output = dy.concatenate([rnn_output, brown_feature_vector[i]])
					elif self.args.use_tag_features:
						rnn_output = dy.concatenate([rnn_output, tag_feature_vector[i]])
					i+=1

					rnn_output = dy.dropout(rnn_output, self.args.dropout)
					probs, rnn_prob = self._get_probs(rnn_output,k)
				p.append(probs)
				rnn_p.append(rnn_prob)

			all_probs.append(p)
			rnn_probs.append(rnn_p)

		if self.args.no_ensemble:
			loss=[]
			for i in range(len(all_probs[0])):
				p = []
				for k in range(self.num_basis):
					if self.args.use_hinge_loss:
						p.append(rnn_probs[k][i])
					else:
						p.append(all_probs[k][i])
				probs = dy.esum(p)
				if self.args.use_hinge_loss:
					loss.append(dy.hinge(probs, output_sentence[i]))
				else:
					loss.append(-dy.log(dy.pick(probs, output_sentence[i])))
			
			loss=dy.esum(loss)

			if self.args.use_regularization:
				if return_val:
					return loss + self.regularize_weights(),0
				else:
					return loss + self.regularize_weights()
			else:
				if return_val:
					return loss,0
				else:
					return loss
		else:
			if pretrain:
				all_loss=[]
				for k in range(self.num_basis):
					loss=[]
					for i in range(len(all_probs[0])):
						if self.args.use_hinge_loss:
							loss.append(dy.hinge(rnn_probs[k][i], output_sentence[i]))
						else:
							loss.append(-dy.log(dy.pick(all_probs[k][i], output_sentence[i])))
					all_loss.append(ensemble_weights[k]*dy.esum(loss))
				loss = dy.esum(all_loss)

				if self.args.use_regularization:
					if return_val:
						return loss + self.regularize_weights(pretrain),aa.index(max(aa))
					else:
						return loss + self.regularize_weights(pretrain)
				else:
					if return_val:
						return loss,aa.index(max(aa))
					else:
						return loss

			else:
				loss=[]
				for i in range(len(all_probs[0])):
					p = []
					for k in range(self.num_basis):
						if self.args.use_hinge_loss:
							p.append(rnn_probs[k][i]*ensemble_weights[k])
						else:
							p.append(all_probs[k][i]*ensemble_weights[k])
					probs = dy.esum(p)
					if self.args.use_hinge_loss:
						loss.append(dy.hinge(probs, output_sentence[i]))
					else:
						loss.append(-dy.log(dy.pick(probs, output_sentence[i])))
				loss=dy.esum(loss)

				if self.args.use_regularization:
					if return_val:
						return loss + self.regularize_weights(), aa.index(max(aa))
					else:
						return loss + self.regularize_weights()
				else:
					if return_val:
						return loss, aa.index(max(aa))
					else:
						return loss
		

	def full_evaluate_acc(self, input_sentence, original_input, true_output, author_vec, author_id, pretrain):
		if self.args.use_all_networks:
			if author_vec[0]:
				author_vec1 = dy.inputTensor(author_vec[0])
			else:
				author_vec1 = dy.lookup(self.non_author_vector, 0)

			if author_vec[1]:
				author_vec2 = dy.inputTensor(author_vec[1])
			else:
				author_vec2 = dy.lookup(self.non_author_vector, 1)

			if author_vec[2]:
				author_vec3 = dy.inputTensor(author_vec[2])
			else:
				author_vec3 = dy.lookup(self.non_author_vector, 2)

			author_vec = dy.concatenate([author_vec1,author_vec2,author_vec3])

		else:
			if author_vec:
				author_vec = dy.inputTensor(author_vec)
			else:
				author_vec = dy.lookup(self.non_author_vector, 0)

		x_recon, z_mu, z_logvar = self.vae_forward(author_vec)
		z, vae_loss, l2_loss = self.vae_loss(author_vec, x_recon, z_mu, z_logvar, kld_factor=0.1)

		correct, total, oov, wrong, correct2, eid = self.evaluate_acc(input_sentence, original_input, true_output, z, author_id, pretrain)
		
		return (correct, total, oov, wrong, correct2, eid)


	
	def evaluate_acc(self, input_sentence, original_input, true_output, author_vec, author_id, pretrain):
		generated_output , eid, eid2 = self.generate(input_sentence, original_input, author_vec, pretrain)
		correct = 0
		total = 0
		count=0
		wrong=0
		oov=0
		correct2=0
		if self.args.write_errors:
			f=open(self.args.log_errors_file,'a')
			wf = open(self.args.log_errors_file[:-4]+'22.log','a')
			wf.write(author_id+'\t')
			wf.write(str(eid2[0])+'\t'+str(eid2[1])+'\t'+str(eid2[2])+'\n')

			f.write(author_id+'\t')
			for g,t in zip(generated_output, true_output):
				f.write(original_input[count]+'\t'+str(g) + '\t'+str(t)+'\t')
				if g==t:
					correct+=1
					if input_sentence[count] not in self.word_to_ix:
						correct2+=1
				else:
					
					#f.write('eid: '+str(eid2[0])+' '+str(eid2[1])+' '+str(eid2[2])+'\n')
					wrong+=1
					if input_sentence[count] not in self.word_to_ix:
						oov+=1
				count+=1
				total+=1
			f.write('\n')
			f.close()
			wf.close()
		else:

			for g,t in zip(generated_output, true_output):
				if g==t:
					correct+=1
					if input_sentence[count] not in self.word_to_ix:
						correct2+=1
				else:
					wrong+=1
					if input_sentence[count] not in self.word_to_ix:
						oov+=1
				count+=1
				total+=1
		return (correct, total, oov, wrong, correct2, eid)

	def evaluate_acc2(self, input_sentence, original_input, true_output, author_vec, pretrain):

		confusion_matrix = np.zeros((len(self.tag_to_ix),len(self.tag_to_ix)))
		generated_output , eid = self.generate(input_sentence, original_input, author_vec, pretrain)

		correct = 0
		wrong = 0
		total = 0
		for g,t in zip(generated_output, true_output):
			confusion_matrix[self.tag_to_ix[t]][self.tag_to_ix[g]]+=1
			if g==t:
				correct+=1
			else:
				wrong+=1
			total+=1

		return (correct, total, confusion_matrix, eid)

	# given an input sentence, run the model and gets the predicted output
	def generate(self, input_sentence, original_input, author_vec, pretrain):

		#dy.renew_cg()

		author_vec_check = 1

		if self.args.use_brown_feats and self.args.use_tag_features:
			surface_feature_vector = self._construct_feature(input_sentence)
		elif self.args.use_brown_feats:
			brown_feature_vector = self._construct_feature1(input_sentence)
		elif self.args.use_tag_features:
			tag_feature_vector = self._construct_feature2(input_sentence)

		#author_vec = dy.vecInput(self.args.node_dim).set(author_vec)
		if not self.args.use_vae:
			if self.args.use_all_networks:
				if author_vec[0]:
					author_vec1 = dy.inputTensor(author_vec[0])
				else:
					author_vec1 = dy.lookup(self.non_author_vector, 0)

				if author_vec[1]:
					author_vec2 = dy.inputTensor(author_vec[1])
				else:
					author_vec2 = dy.lookup(self.non_author_vector, 1)

				if author_vec[2]:
					author_vec3 = dy.inputTensor(author_vec[2])
				else:
					author_vec3 = dy.lookup(self.non_author_vector, 2)

				author_vec = dy.concatenate([author_vec1,author_vec2,author_vec3])

			else:
				if author_vec:
					author_vec = dy.inputTensor(author_vec)

					if self.args.dont_use_author_vec:
						author_vec_check = 1
					if self.args.just_ensemble:
						author_vec_check = None

				else:
					author_vec = dy.lookup(self.non_author_vector, 0)

					if self.args.dont_use_author_vec or self.args.just_ensemble:
						author_vec_check = None

		#if author_vec:
		#	author_vec = dy.inputTensor(author_vec)
		#else:
		#	author_vec = dy.lookup(self.non_author_vector, 0)


		if self.args.no_ensemble or self.args.just_ensemble:
			pass
		else:
			if pretrain:
				
				rand_basis_vecs = [dy.lookup(self.random_basis_vectors, k, update=False) for k in range(self.num_basis)]
				if self.args.only_pretrain:
					ensemble_weights = dy.logistic(dy.concatenate([dy.dot_product(author_vec,vec) for vec in rand_basis_vecs]))
					aa = ensemble_weights.value()
				else:
					ensemble_weights = [dy.logistic(dy.dot_product(author_vec,vec)) for vec in rand_basis_vecs]
					aa = [ew.value() for ew in ensemble_weights]
				
			else:
				basis_mx = dy.parameter(self.basis_matrix)
				basis_bs = dy.parameter(self.basis_bias)
				if self.args.no_bias:
					if self.args.use_logistic:
						ensemble_weights = dy.logistic(basis_mx*author_vec)
					else:
						ensemble_weights = dy.softmax(basis_mx*author_vec)
				else:
					if self.args.use_logistic:
						ensemble_weights = dy.logistic(basis_mx*author_vec + basis_bs)
					else:
						ensemble_weights = dy.softmax(basis_mx*author_vec+ basis_bs)

				aa = ensemble_weights.value()

		if self.args.dont_use_author_vec or self.args.just_ensemble:
			v=[]
			for ii in range(self.num_basis):
				v.append(1/self.num_basis)

			if not author_vec_check:
				ensemble_weights = dy.inputTensor(v)
				aa = ensemble_weights.value()

		input_input = input_sentence
		input_sentence = self._preprocess_input(input_sentence, self.word_to_ix)

		for i in range(self.num_basis):
			self.char_bi_lstm[i].disable_dropout()
			self.word_bi_lstm[i].disable_dropout()

		all_probs=[]
		rnn_probs=[]
		for k in range(self.num_basis):
			kk=k

			word_embeds=[]
			for i in range(len(original_input)):
				word = original_input[i]
				chars = ['<-s->'] + list(word) + ['<-e->']
				input_chars = self._preprocess_input(chars, self.char_to_ix)
				embed_chars = self._embed_word(input_chars,k)
				#embed_chars = [dy.dropout(embed, self.args.dropout) for embed in embed_chars]
				k=kk
				if self.args.freeze_char_lstm:
					k=0
				char_rnn_states = self._run_rnn2(self.char_bi_lstm[k], embed_chars)
				word_embed = dy.concatenate([char_rnn_states[1][1], char_rnn_states[-2][0]])

				char_output_w = dy.parameter(self.char_output_w[k])
				char_output_b = dy.parameter(self.char_output_b[k])
				char_output_w2 = dy.parameter(self.char_output_w2[k])
				char_output_b2 = dy.parameter(self.char_output_b2[k])
				word_embed = dy.tanh(char_output_w * word_embed + char_output_b)
				word_embed = char_output_w2 * word_embed + char_output_b2
				word_embeds.append(word_embed)

			if self.args.use_pretrained_embed and self.args.learn_word_embed:
				embedded_sentence = self._embed_sentence(input_sentence,k)
				input_embed = [dy.inputTensor(self.word_vecs[word]) if word in self.word_vecs else dy.inputTensor(self.word_vecs['unk']) for word in input_input ]
				embeds = [dy.concatenate([i,j,k]) for i,j,k in zip(word_embeds,embedded_sentence, input_embed)]

			elif self.args.use_pretrained_embed==False and self.args.learn_word_embed:
				embedded_sentence = self._embed_sentence(input_sentence,k)
				embeds = [dy.concatenate([i,j]) for i,j in zip(word_embeds,embedded_sentence)]

			elif self.args.use_pretrained_embed and self.args.learn_word_embed==False:
				input_embed = [dy.inputTensor(self.word_vecs[word]) if word in self.word_vecs else dy.inputTensor(self.word_vecs['unk']) for word in input_input ]
				embeds = [dy.concatenate([i,k]) for i,k in zip(word_embeds, input_embed)]

			else:
				embeds = word_embeds

			#embeds = [dy.dropout(embed, self.args.dropout) for embed in embeds]

			k=kk
			if self.args.freeze_word_lstm:
				k=0
			rnn_outputs = self._run_rnn(self.word_bi_lstm[k], embeds)


			p=[]
			rnn_p=[]
			i=0
			for rnn_output in rnn_outputs:

				if self.args.use_node_feature:
					rnn_output = dy.concatenate([rnn_output, author_vec])

				if self.args.freeze_surface_features:
					if self.args.use_tag_features and self.args.use_brown_feats:
						probs, rnn_prob = self._get_surface_probs(rnn_output, k, surface_feature_vector[i])
					elif self.args.use_brown_feats:
						probs, rnn_prob = self._get_surface_probs(rnn_output, k, brown_feature_vector[i])
					elif self.args.use_tag_features:
						probs, rnn_prob = self._get_surface_probs(rnn_output, k, tag_feature_vector[i])
					i+=1
				else:	
					if self.args.use_tag_features and self.args.use_brown_feats:
						rnn_output = dy.concatenate([rnn_output, surface_feature_vector[i]])
					elif self.args.use_brown_feats:
						rnn_output = dy.concatenate([rnn_output, brown_feature_vector[i]])
					elif self.args.use_tag_features:
						rnn_output = dy.concatenate([rnn_output, tag_feature_vector[i]])
					i+=1

					probs, rnn_prob = self._get_probs(rnn_output,k)
				p.append(probs)
				rnn_p.append(rnn_prob)

			all_probs.append(p)
			rnn_probs.append(rnn_p)

		if self.args.no_ensemble:
			output_sentence=[]
			for i in range(len(all_probs[0])):
				p = []
				for k in range(self.num_basis):
					if self.args.use_hinge_loss:
						p.append(rnn_probs[k][i])
					else:
						p.append(all_probs[k][i])
				probs = dy.esum(p)
				predicted_word = self._predict(probs)
				output_sentence.append(predicted_word)

			return output_sentence,0
		else:

			# need to add share previous tag
			output_sentence=[]
			for i in range(len(all_probs[0])):
				p = []
				for k in range(self.num_basis):
					if self.args.use_hinge_loss:
						p.append(rnn_probs[k][i]*ensemble_weights[k])
					else:
						p.append(all_probs[k][i]*ensemble_weights[k])
				probs = dy.esum(p)
				predicted_word = self._predict(probs)
				output_sentence.append(predicted_word)

			return output_sentence,aa.index(max(aa)),aa

	






























