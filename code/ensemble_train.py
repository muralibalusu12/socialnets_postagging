import dynet_config
#dynet_config.set_gpu()
#dynet_config.set(random_seed=1)
import dynet as dy
import argparse
import logging
import os
import random
from collections import Counter, defaultdict
#import matplotlib.pyplot as plt
import numpy as np

import arguments as arguments
import ensemble_model as model
#import constants as constants

import sklearn
from scipy import stats
#import seaborn as sn
#import pandas as pd
#import matplotlib.pyplot as plt
import re

import emoji
import pickle
import time

from sklearn.preprocessing import normalize

def preprocess_token(word):
	def OR_exp(exp):
		prefix = "(?: "
		final_exp = ''
		for e in exp:
			final_exp+=prefix
			prefix=' | '
			final_exp+=e
		final_exp+=" )"
		return final_exp

	Contractions = re.compile("(?i)(\\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$");
	Whitespace = re.compile("[\s]+\Z\s");  #[\\s\\\p{Zs}]+

	punctChars = "['\"“”‘’.?!…,:;]";
	punctSeq   = "['\"“”‘’]+|[.?!,…]+|[:;]+";
	entity     = "&(?:amp|lt|gt|quot);";

	urlStart1  = "(?:https?://|\\bwww\\.)";
	commonTLDs = "(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)";
	ccTLDs = "(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" +\
				"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" +\
				"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" +\
				"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" +\
				"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" +\
				"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" +\
				"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" +\
				"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)";
	urlStart2  = "\\b(?:[A-Za-z\\d-])+(?:\\.[A-Za-z0-9]+){0,3}\\." + "(?:"+commonTLDs+"|"+ccTLDs+")"+\
				"(?:\\."+ccTLDs+")?(?=\\W|$)";
	urlBody    = "(?:[^\\.\\s<>][^\\s<>]*?)?";
	urlExtraCrapBeforeEnd = "(?:"+punctChars+"|"+entity+")+?";
	urlEnd     = "(?:\\.\\.+|[<>]|\\s|$)";
	url        = "(?:"+urlStart1+"|"+urlStart2+")"+urlBody+"(?=(?:"+urlExtraCrapBeforeEnd+")?"+urlEnd+")";


	# Emoticons
	normalEyes = "(?iu)[:=]"; 
	wink = "[;]";
	noseArea = "(?:|-|[^a-zA-Z0-9 ])"; 
	happyMouths = "[D\\)\\]\\}]+";
	sadMouths = "[\\(\\[\\{]+";
	tongue = "[pPd3]+";
	otherMouths = "(?:[oO]+|[/\\\\]+|[vV]+|[Ss]+|[|]+)"; 
	bfLeft = "(♥|0|o|°|v|\\$|t|x|;|\\u0CA0|@|ʘ|•|・|◕|\\^|¬|\\*)";
	bfCenter = "(?:[\\.]|[_-]+)";
	bfRight = "\\2";
	s3 = "(?:--['\"])";
	s4 = "(?:<|&lt;|>|&gt;)[\\._-]+(?:<|&lt;|>|&gt;)";
	s5 = "(?:[.][_]+[.])";
	basicface = "(?:(?i)" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5;

	eeLeft = "[＼\\\\ƪԄ\\(（<>;ヽ\\-=~\\*]+";
	eeRight= "[\\-=\\);'\\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+";
	eeSymbol = "[^A-Za-z0-9\\s\\(\\)\\*:=-]";
	eastEmote = eeLeft + "(?:"+basicface+"|" +eeSymbol+")+" + eeRight;
	emoticon = OR_exp(["(?:>|&gt;)?" + OR_exp([normalEyes, wink]) + OR_exp([noseArea,"[Oo]"]) +\
                   OR_exp([tongue+"(?=\\W|$|RT|rt|Rt)", otherMouths+"(?=\\W|$|RT|rt|Rt)", sadMouths, happyMouths]),\
                   "(?<=(?:|^))" + OR_exp([sadMouths,happyMouths,otherMouths]) + noseArea +\
                   OR_exp([normalEyes, wink]) + "(?:<|&lt;)?", re.sub("2","1",eastEmote,1),basicface]);

	Hearts = "(?:<+/?3+)+";

	Arrows = "(?:<*[-―—=]*>+|<+[-―—=]*>*)"; ##(?:<*[-―—=]*>+|<+[-―—=]*>*)|\\p{InArrows}+

	Hashtag = "#[a-zA-Z0-9_]+";

	AtMention = "[@＠][a-zA-Z0-9_]+"; 

	Bound = "(?:^|$)";  ##(?:\\W|^|$)
	Email = "(?<=" +Bound+ ")[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,4}(?=" +Bound+")";

	if re.search(Email, word):
		#base=''
		#m = re.search(justbase, word)
		#if m:
		#	base=m.group().lower()
		#	#word = '<Email-'+base+'>'
		word = '<url>'
	elif re.findall(url, word):
		#base=''
		#m = re.search(justbase, word)
		#if m:
		#	base=m.group().lower()
		#	#word = '<URL-'+base+'>'
		word = '<url>'
	elif re.match(AtMention,word):
		word = '<user>'
	elif re.match(Hashtag,word):
		word = '<hashtag>'
	elif re.search(emoticon,word):
		word = '<emoticon>'
	elif re.search(Arrows,word):
		word = '<arrow>'
	elif re.search(Hearts,word):
		word = '<heart>'
	else:
		numbers = re.findall('[-+]?[.\d]*[\d]+[:,.\d]*',word)
		if len(numbers)>0:
			word='<number>'
		
		word = word.lower()

	return word

def parse_file(filename, dont_preprocess_data, dont_lowercase_words, share_hiddenstates, just_pad_sents):
	if not dont_lowercase_words:
		logging.info('lowercasing words in the dataset')
	if not dont_preprocess_data:
		logging.info('preprocessing dataset')
	if share_hiddenstates or just_pad_sents:
		logging.info('adding start and end append words to share hidden states later')
	data = []
	count = 0			#maintaining a count if needed
	with open(filename) as f:
		sentence=[]
		labels=[]
		for i,line in enumerate(f):
			if line.strip()=='':
				assert (len(sentence)==len(labels))
				if share_hiddenstates or just_pad_sents:
					sentence = ['<start>'] + sentence + ['<end>']
				sentence = sentence 
				labels = labels
				data.append((sentence,labels))
				sentence=[]
				labels=[]
			else:
				word,tag = line.split('\t')
				if dont_lowercase_words:
					pass
				else:
					word = word.lower()

				if dont_preprocess_data:
					pass
				else:
					word = preprocess_token(word)
					'''
					dollars = re.findall('[^\s]*\$[^\s]*',word)
					if len(dollars)>0:
						word='<$>'
						count+=1
					urls = re.findall('https?://[^\s]*',word)
					if len(urls)>0:
						word='<url>'
						count+=1
					if word in emoji.UNICODE_EMOJI:
						word='emoticon'
						count+=1
					if word[0]=='#':
						word='<hashtag>'
						count+=1
					if word[0]=='@':
						word='<user>'
						count+=1
					numbers = re.findall('[-+]?[.\d]*[\d]+[:,.\d]*',word)
					if len(numbers)>0:
						word='<number>'
						count+=1
					#'''
				sentence.append(word)
				labels.append(tag.strip())
	return data


def get_word_label_ix(train_file, vocab_size):
	char_to_ix = {'unk':0,'<-s->':1,'<-e->':2}
	word_dict=Counter()
	label_to_ix={}
	#label_to_ix={'unk':0}

	for sentence,labels in train_file:
		for word in sentence:
			word_dict[word] += 1
			for char in word:
				if char not in char_to_ix:
					char_to_ix[char] = len(char_to_ix)
		for label in labels:
			if label not in label_to_ix:
				label_to_ix[label] = len(label_to_ix)

	logging.info('total no of words: %d',len(word_dict))
	top_words = word_dict.most_common(vocab_size-1)

	logging.info('reducing no of words to vocab_size: %d',vocab_size)

	word_to_ix={'unk':0 }#,'<start>':1,'<end>':2}
	for e in top_words:
		val,count = e
		if val not in word_to_ix:
			word_to_ix[val]=len(word_to_ix)

	logging.info('reduced no of words to vocab_size: %d',len(word_to_ix))
	
	return char_to_ix, word_to_ix, label_to_ix


def extract_embeds(file_name, embed_dim, word_to_ix):
	word_vecs = {}

	#'''
	with open('data/glove.twitter.27B/glove.twitter.27B.100d.txt','r') as f:
		for i,line in enumerate(f):
			if len(line.split())>embed_dim:
				word = line.split()[0]
				vec = [float(v) for v in line.split()[1:]]
				if word in word_to_ix:
					word_vecs[word] = vec
	#'''
	

	logging.info('no of actual glove embeddings used: '+str(len(word_vecs)))
	
	if 'unk' not in word_vecs:
		word_vecs['unk'] = [0]*embed_dim # adding 'unk' token to this if not there

	return word_vecs


def extract_authorvecs2(filename):
	author_vecs={} 
	with open(filename,'r') as f:
		for i,line in enumerate(f):
			content = line.strip().split()[1:]
			vec = [float(v) for v in content]
			#if np.linalg.norm(vec)!=0:
			#	author_vecs[line.strip().split()[0]]=list(np.array(vec)/np.linalg.norm(vec))
			#else:
			author_vecs[line.strip().split()[0]] = vec
	return author_vecs

def extract_authorvecs(args):
	if args.network=='follow':
		logging.info('extracting follow embeddings')
		author_vecs = extract_authorvecs2(args.follow_vecs)
	elif args.network=='mention':
		logging.info('extracting mention embeddings')
		author_vecs = extract_authorvecs2(args.mention_vecs)
	elif args.network=='retweet':
		logging.info('extracting retweet embeddings')
		author_vecs = extract_authorvecs2(args.retweet_vecs)

	
	random_vec = None
	train_author_vecs=[]
	train_author_ids=[]
	with open(args.trainid_data,'r') as f:
		for i,line in enumerate(f):
			author = line.strip().split()[0]
			train_author_ids.append(author)
			if author in author_vecs:
				train_author_vecs.append(author_vecs[author])
			else:
				train_author_vecs.append(random_vec)

	dev_author_vecs=[]
	dev_author_ids=[]
	with open(args.devid_data,'r') as f:
		for i,line in enumerate(f):
			author = line.strip().split()[0]
			dev_author_ids.append(author)
			if author in author_vecs:
				dev_author_vecs.append(author_vecs[author])
				if args.combine_train_dev or args.combine_train_dev_test:
					train_author_vecs.append(author_vecs[author])
					train_author_ids.append(author)
			else:
				dev_author_vecs.append(random_vec)
				if args.combine_train_dev or args.combine_train_dev_test:
					train_author_vecs.append(random_vec)
					train_author_ids.append(author)

	test_author_vecs=[]
	test_author_ids=[]
	with open(args.testid_data,'r') as f:
		for i,line in enumerate(f):
			author = line.strip().split()[0]
			test_author_ids.append(author)
			if author in author_vecs:
				test_author_vecs.append(author_vecs[author])
				if args.combine_train_dev_test:
					train_author_vecs.append(author_vecs[author])
					train_author_ids.append(author)
			else:
				test_author_vecs.append(random_vec)
				if args.combine_train_dev_test:
					train_author_vecs.append(random_vec)
					train_author_ids.append(author)

	test2_author_vecs=[]
	test2_author_ids=[]
	with open(args.testid_data2,'r') as f:
		for i,line in enumerate(f):
			author = line.strip().split()[0]
			test2_author_ids.append(author)
			if author in author_vecs:
				test2_author_vecs.append(author_vecs[author])
			else:
				test2_author_vecs.append(random_vec)

	logging.info('len(train_author_vecs): '+str(len(train_author_vecs)))
	logging.info('len(dev_author_vecs): '+str(len(dev_author_vecs)))
	logging.info('len(test_author_vecs): '+str(len(test_author_vecs)))
	logging.info('len(test2_author_vecs): '+str(len(test2_author_vecs)))

	logging.info('len(train_author_ids): '+str(len(train_author_ids)))
	logging.info('len(dev_author_ids): '+str(len(dev_author_ids)))
	logging.info('len(test_author_ids): '+str(len(test_author_ids)))
	logging.info('len(test2_author_ids): '+str(len(test2_author_ids)))


	return train_author_vecs, dev_author_vecs, test_author_vecs, test2_author_vecs, train_author_ids, dev_author_ids, test_author_ids, test2_author_ids

'''
def extract_embeds2(file_name, embed_dim, word_to_ix):
	word_vecs2 = {}
	with open(filename) as f:
		for i,line in enumerate(f):
			if len(line.split())>embed_dim:
				word = line.split()[0]
				vec = [float(v) for v in line.split()[1:]]
				if word in word_to_ix:
					word_vecs2[word] = vec

	if 'unk' not in word_vecs2:
		word_vecs2['unk'] = [0]*embed_dim # adding 'unk' token to this if not there
	word_vectors2 = [word_vecs2[word] if word in word_vecs2 else word_vecs2['unk'] for word,val in word_to_ix.items()]
	return word_vectors
'''



def char_train(network, train_set, val_set, test_set, test_set2, train_set_word, val_set_word,
					test_set_word, test_set2_word, epochs, batch_size, args, tag_to_ix):
	
	def get_val_set_loss(network, val_set,val_set_word, val_author_vecs, pretrain, num_basis):
		loss = []
		vae_loss=[0]
		l2_loss=[0]
		for i,(input_sentence, output_sentence) in enumerate(val_set):
			if args.use_vae:
				l,a,v,l2 = network.get_full_loss(input_sentence, val_set_word[i][0], output_sentence, val_author_vecs[i], pretrain)
				loss.append(l.value())
				vae_loss.append(v.value())
				l2_loss.append(l2.value())
			else:
				loss.append(network.get_loss(input_sentence, val_set_word[i][0], output_sentence, val_author_vecs[i], pretrain).value())
			dy.renew_cg()
		return sum(loss)/len(val_set), sum(vae_loss)/len(val_set), sum(l2_loss)/len(val_set)

	def get_val_set_acc(network, val_set, val_set_word, val_author_vecs, val_author_ids, pretrain, num_basis):
		evals=[]
		if args.use_vae:
			for i, (input_sentence, output_sentence) in enumerate(val_set):
				evals.append(network.full_evaluate_acc(input_sentence, val_set_word[i][0], output_sentence, val_author_vecs[i], val_author_ids[i], pretrain))
				dy.renew_cg()
		else:
			for i, (input_sentence, output_sentence) in enumerate(val_set):
				evals.append(network.evaluate_acc(input_sentence, val_set_word[i][0], output_sentence, val_author_vecs[i], val_author_ids[i], pretrain))
				dy.renew_cg()
		dy.renew_cg()

		correct = [c for c,t,d,w,cc,e in evals]
		total = [t for c,t,d,w,cc,e in evals]
		mean=0
		confidence=0
		oov = [d for c,t,d,w,cc,e in evals]
		wrong = [w for c,t,d,w,cc,e in evals]
		correct2 = [cc for c,t,d,w,cc,e in evals]

		auth_correct = [c for i,(c,t,d,w,cc,e) in enumerate(evals) if val_author_vecs[i] is not None ]
		auth_total = [t for i,(c,t,d,w,cc,e) in enumerate(evals) if val_author_vecs[i] is not None ]
		non_auth_correct = [c for i,(c,t,d,w,cc,e) in enumerate(evals) if val_author_vecs[i] is None ]
		non_auth_total = [t for i,(c,t,d,w,cc,e) in enumerate(evals) if val_author_vecs[i] is None ]
		eids = [ e for c,t,d,w,cc,e in evals]
		#unique_eid = set(eids)
		len_eid = num_basis
		counts = []
		for i in range(len_eid):
			counts.append(sum([e==i for e in eids]))
		counts2 = []
		for i in range(len_eid):
			counts2.append(sum([e==i for j,e in enumerate(eids) if val_author_vecs[j] is not None ]))

		if sum(non_auth_total)==0:
			non_auth_total=[1]

		return 100.0*sum(correct)/sum(total), mean, confidence, sum(oov), sum(wrong), sum(correct2), 100.0*sum(auth_correct)/sum(auth_total), 100.0*sum(non_auth_correct)/sum(non_auth_total), counts, counts2
	
	#original_set = train_set 
	#train_set = train_set*epochs

	if args.optimizer=='adadelta':
		trainer = dy.AdadeltaTrainer(network.model)
		trainer.set_clip_threshold(5)
	elif args.optimizer=='adam':
		trainer = dy.AdamTrainer(network.model, alpha = args.lr)
		trainer.set_clip_threshold(5)
	elif args.optimizer=='sgd-momentum':
		trainer = dy.MomentumSGDTrainer(network.model, learning_rate = args.lr)
	else:
		logging.critical('This Optimizer is not valid or not allowed')

	losses = []
	iterations = []

	kk=args.pretrain_epochs

	if args.use_all_networks:
		args.network='follow'
		train_author_vecs1, dev_author_vecs1, test_author_vecs1, test2_author_vecs1, train_author_ids, dev_author_ids, test_author_ids, test2_author_ids = extract_authorvecs(args)

		args.network='mention'
		train_author_vecs2, dev_author_vecs2, test_author_vecs2, test2_author_vecs2, _,_,_,_ = extract_authorvecs(args)

		args.network='retweet'
		train_author_vecs3, dev_author_vecs3, test_author_vecs3, test2_author_vecs3, _,_,_,_ = extract_authorvecs(args)

		train_author_vecs=[]
		for i,j,k in zip(train_author_vecs1,train_author_vecs2,train_author_vecs3):
			train_author_vecs.append((i,j,k))

		dev_author_vecs=[]
		for i,j,k in zip(dev_author_vecs1,dev_author_vecs2,dev_author_vecs3):
			dev_author_vecs.append((i,j,k))

		test_author_vecs=[]
		for i,j,k in zip(test_author_vecs1,test_author_vecs2,test_author_vecs3):
			test_author_vecs.append((i,j,k))

		test2_author_vecs=[]
		for i,j,k in zip(test2_author_vecs1,test2_author_vecs2,test2_author_vecs3):
			test2_author_vecs.append((i,j,k))

	else:
		train_author_vecs, dev_author_vecs, test_author_vecs, test2_author_vecs, train_author_ids, dev_author_ids, test_author_ids, test2_author_ids = extract_authorvecs(args)

	logging.info('obtained all author vectors '+str(len(train_author_vecs))+' '+str(len(dev_author_vecs))+' '+str(len(test_author_vecs))+' '+str(len(test2_author_vecs)))

	batch_loss_vec=[]
	dy.renew_cg()

	is_best = 0
	best_val = 0
	count = 0
	count_train=-1

	#early_stopping = 0

	for epoch in range(epochs):
		#if early_stopping>args.early_epochs:
		#	break

		all_inds=[]
		num_train = int(len(train_set)/args.batch_size + 1)*args.batch_size

		#prev_time=time.time()

		for ii in range(num_train):
			
			count_train+=1
			if count_train==len(train_set):
				count_train=0


			count+=1
			inputs, outputs = train_set[count_train]
			inputs_word,_ = train_set_word[count_train]

			'''
			data_point = {'inputs':inputs, 'inputs_word':inputs_word, 'outputs':outputs, 'train_author_vecs':train_author_vecs[i]}
			pickle.dump(data_point,open( "data_pickle/"+str(i)+".p", "wb" ))
			data_point = pickle.load( open( "data_pickle/"+str(i)+".p", "rb" ) )
			inputs = data_point['inputs']
			inputs_word = data_point['inputs_word']
			outputs = data_point['outputs']
			train_author_vec = data_point['train_author_vecs']
			'''

			#prev_time2 = time.time()
			#if train_author_vecs[count_train] !=None:

			vae_loss=0
			if args.use_vae:
				loss,ind,vae_loss,l2_loss = network.get_full_loss(inputs, inputs_word, outputs, train_author_vecs[count_train], epoch<kk, True)
			else:
				loss,ind = network.get_loss(inputs, inputs_word, outputs, train_author_vecs[count_train], epoch<kk, True)

			#curr_time2 = time.time()

			#print ('time for one instance: ', curr_time2 - prev_time2)

			all_inds.append(ind)
			#print (loss)
			#a = input()
			batch_loss_vec.append(loss)

			if count%batch_size==0:
				
				batch_loss = dy.esum(batch_loss_vec)/batch_size
				batch_loss.forward()
				batch_loss.backward()
				trainer.update()
				batch_loss_vec=[]
				dy.renew_cg()
				count=0
			#logging.info('finished minibatch: %d/%d',ii,num_train)
				

		#print ('until here-----')
		#curr_time = time.time()
		#print ('time for one epoch training: ', curr_time - prev_time)

		

		counts = []
		for i in range(args.num_basis):
			a = [v==i for v in all_inds]
			counts.append(sum(a))
		logging.info('distribution of the data points'+str(counts))

		#if ((i+1))%len(original_set) == 0:
		if args.plots:
			val_loss = get_val_set_loss(network, val_set, val_set_word, dev_author_vecs,epoch<kk, args.num_basis)
			losses.append(val_loss)
			iterations.append(epoch)
		#dy.renew_cg()
		
		#if ((i+1))%len(original_set)==0:
		train_loss=0
		if args.slow:
			train_loss,train_vae_loss,train_l2_loss = get_val_set_loss(network, train_set, train_set_word, train_author_vecs,epoch<kk, args.num_basis)
		
		if args.write_errors:
			f=open(args.log_errors_file,'a')
			f.write('\n--------- epoch no: --------- ')
			f.write(str(epoch) +'\n')
			f.close()
			f=open(args.log_errors_file,'a')
			f.write('\n--------- oct27.train errors: --------- \n')
			f.close()
		#prev_time = time.time()
		trainacc, train_acc, train_confidence, oov_train, wrong_train, correct_train, auth_acc1, non_auth_acc1, eids1, counts21 = get_val_set_acc(network, train_set, train_set_word, train_author_vecs, train_author_ids, epoch<kk, args.num_basis)
		#curr_time = time.time()
		#print ('time for acc train: ', curr_time - prev_time)

		if args.write_errors:
			f=open(args.log_errors_file,'a')
			f.write('\n--------- oct27.dev errors: ---------\n')
			f.close()
		
		val_loss,val_vae_loss,val_l2_loss = 0,0,0
		val_acc, oov_val, wrong_val, correct_val = 0,0,0,0

		if args.slow:
			pass
			#val_loss,val_vae_loss = get_val_set_loss(network, val_set, val_set_word, dev_author_vecs,epoch<kk, args.num_basis)
		#prev_time = time.time()
		valacc, val_acc, val_confidence, oov_val, wrong_val, correct_val, auth_acc2, non_auth_acc2, eids2, counts22=0,0,0,0,0,0,0,0,0,0
		#valacc, val_acc, val_confidence, oov_val, wrong_val, correct_val, auth_acc2, non_auth_acc2, eids2, counts22 = get_val_set_acc(network, val_set, val_set_word, dev_author_vecs, dev_author_ids, epoch<kk, args.num_basis)
		#curr_time = time.time()
		#print ('time for acc val: ', curr_time - prev_time)

		if args.write_errors:
			f=open(args.log_errors_file,'a')
			f.write('\n---------  oct27.test errors: --------- \n')
			f.close()
		test_loss=0
		if args.slow:
			test_loss, test_vae_loss, test_l2_loss = get_val_set_loss(network, test_set, test_set_word, test_author_vecs,epoch<kk, args.num_basis)
		#prev_time = time.time()
		testacc, test_acc, test_confidence, oov_test, wrong_test, correct_test, auth_acc3, non_auth_acc3, eids3, counts23 = get_val_set_acc(network, test_set, test_set_word, test_author_vecs, test_author_ids, epoch<kk, args.num_basis)
		#curr_time = time.time()
		#print ('time for acc test: ', curr_time - prev_time)

		if args.write_errors:
			f=open(args.log_errors_file,'a')
			f.write('\n---------  daily547.test errors: --------- \n')
			f.close()
		test_loss2=0
		if args.slow:
			test_loss2, test_vae_loss2, test2_l2_loss = get_val_set_loss(network, test_set2, test_set2_word, test2_author_vecs,epoch<kk, args.num_basis)
		#prev_time = time.time()
		testacc2, test_acc2, test2_confidence, oov_test2, wrong_test2, correct_test2, auth_acc4, non_auth_acc4, eids4, counts24 = get_val_set_acc(network, test_set2, test_set2_word, test2_author_vecs, test2_author_ids, epoch<kk, args.num_basis)
		#curr_time = time.time()
		#print ('time for acc test2: ', curr_time - prev_time)

		#test_loss2 = get_val_set_loss(network, test_set2, test_set2_word, test_author_vecs, epoch<kk)
		#test_acc2, oov_test2, wrong_test2, correct_test2, auth_acc4, non_auth_acc4, eids4 = get_val_set_acc(network, test_set2, test_set2_word, test_author_vecs,epoch<kk)

		#prev_time = time.time()
		logging.info('epoch %d done', epoch)
		logging.info('train loss: %f, train vae loss: %f, train l2 loss: %f, train acc: %f', train_loss, train_vae_loss, train_l2_loss, trainacc)
		logging.info('val loss: %f, val vae loss: %f, val l2 loss: %f, val acc: %f', val_loss, val_vae_loss, val_l2_loss, valacc)
		logging.info('test loss: %f, test vae loss: %f, test l2 loss: %f, test acc: %f', test_loss, test_vae_loss, test_l2_loss, testacc)
		logging.info('test2 loss: %f, tes2 vae loss: %f, tes2 l2 loss: %f, test2 acc: %f', test_loss2, test_vae_loss2, test2_l2_loss, testacc2)

		logging.info(' oov_train: %d/%d, %d, oov_val: %d/%d, %d, oov_test: %d/%d, %d, oov_test2: %d/%d, %d', 
			oov_train, wrong_train, correct_train, oov_val, wrong_val, correct_val, oov_test, wrong_test, correct_test, oov_test2, wrong_test2, correct_test2)

		logging.info('train: author_acc: %f, non_author_acc: %f, '+str(eids1)+' '+str(counts21),auth_acc1, non_auth_acc1)
		logging.info('dev: author_acc: %f, non_author_acc: %f, '+str(eids2)+' '+str(counts22),auth_acc2, non_auth_acc2)
		logging.info('test: author_acc: %f, non_author_acc: %f, '+str(eids3)+' '+str(counts23),auth_acc3, non_auth_acc3)
		logging.info('test2: author_acc: %f, non_author_acc: %f, '+str(eids4)+' '+str(counts24),auth_acc4, non_auth_acc4)

		if args.plots:
			test_acc, test_confidence, confusion_matrix, auth_acc, non_auth_acc, eids = get_val_set_acc2(network, test_set, test_set_word, test_author_vecs,epoch<kk, args.num_basis)
			df_cm = pd.DataFrame(confusion_matrix, index = [i for i in tag_to_ix.keys()],
												columns = [i for i in tag_to_ix.keys()])
			fig = plt.figure(figsize = (10,7))
			sn.heatmap(df_cm, annot=True)
			fig.savefig('figs/conf_matrix_'+str(epoch)+'.png')
			#a = input()

		if args.combine_train_dev:
			valacc = testacc
		elif args.combine_train_dev_test:
			valacc = testacc2
		else:
			valacc = valacc

		m = network.model
		if epoch==0:
			best_acc = valacc
			best_epoch = 0
			#best_val = val_loss
			#if args.combine_train_dev:
			#	best_acc = testacc
			#else:
			#	best_acc = valacc
			if args.save_model:
				m.save(args.save_model)
				logging.info('saving best model')
		else:
			#if args.combine_train_dev:
			#	valacc = testacc
			#
			#if best_acc < valacc:
			#	early_stopping = 0
			#	if args.combine_train_dev:
			#		best_acc = testacc
			#	else:
			#		best_acc = valacc
			if best_acc <= valacc:
				best_acc = valacc
				best_epoch = epoch
				if args.save_model:
					m.save(args.save_model)
					logging.info('re-saving best model')
			#else:
			#	early_stopping+=1
		logging.info('best model is at epoch no: %d', best_epoch)

	logging.info('\nbest model details are at epoch no: %d', best_epoch)

		#curr_time = time.time()
		#print ('time for rest junk: ', curr_time - prev_time)
	
	'''
	if count%batch_size!=0:
		batch_loss = dy.esum(batch_loss_vec)/len(batch_loss_vec)
		batch_loss.forward()
		batch_loss.backward()
		trainer.update()
		batch_loss_vec=[]
		dy.renew_cg()
	'''
	

	if args.plots:
		fig = plt.figure()
		plt.plot(iterations, losses)
		axes = plt.gca()
		axes.set_xlim([0,epochs])
		axes.set_ylim([0,10000])

		fig.savefig('figs/loss_plot.png')




def main():

	args = arguments.get_arguments()

	if args.log_file!=None:
		logging.basicConfig(format='%(asctime)s %(message)s', filename=args.log_file, filemode='a', level=logging.DEBUG)
	else:
		logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

	#logging.basicConfig(format='%(asctime)s %(message)s', filename='example.log', filemode='w', level=logging.DEBUG)
	#logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
	#logging.debug('This message should go to the log file')
	#logging.info('So should this')
	#logging.warning('And this, too')

	# Get all the command-line arguments
	#args = arguments.get_arguments()
	logging.info('Obtained all arguments: %s', str(args))

	if args.train_data:
		train_data = parse_file(args.train_data, args.dont_preprocess_data, args.dont_lowercase_words, 
								args.share_hiddenstates, args.just_pad_sents)
	if args.dev_data:
		dev_data = parse_file(args.dev_data, args.dont_preprocess_data, args.dont_lowercase_words, 
								args.share_hiddenstates, args.just_pad_sents)
	if args.test_data:
		test_data = parse_file(args.test_data, args.dont_preprocess_data, args.dont_lowercase_words, 
								args.share_hiddenstates, args.just_pad_sents)
	if args.test_data2:
		test_data2 = parse_file(args.test_data2, args.dont_preprocess_data, args.dont_lowercase_words, 
								args.share_hiddenstates, args.just_pad_sents)
	
	if args.model=='char_bilstm' or args.model=='char_bilstm-crf':
		if args.train_data:
			#train_data_word = train_data
			train_data_word = parse_file(args.train_data, True, True, 
									args.share_hiddenstates, args.just_pad_sents)
			#train_data_word = parse_file2(args.train_data, args.dont_lowercase_words, args.share_hiddenstates)
		if args.dev_data:
			#dev_data_word = dev_data
			dev_data_word = parse_file(args.dev_data, True, True, 
									args.share_hiddenstates, args.just_pad_sents)
			#dev_data_word = parse_file2(args.dev_data, args.dont_lowercase_words, args.share_hiddenstates)
		if args.test_data:
			#test_data_word = test_data
			test_data_word = parse_file(args.test_data, True, True, 
									args.share_hiddenstates, args.just_pad_sents)
			#test_data_word = parse_file2(args.test_data, args.dont_lowercase_words, args.share_hiddenstates)
		if args.test_data2:
			#test_data2_word = test_data2
			test_data2_word = parse_file(args.test_data2, True, True, 
										args.share_hiddenstates, args.just_pad_sents)
			#test_data2_word = parse_file2(args.test_data2, args.dont_lowercase_words, args.share_hiddenstates)
	
	if args.combine_train_dev:
		logging.info('combining training and dev data')
		train_data = train_data + dev_data
		if args.model=='char_bilstm' or args.model=='char_bilstm-crf':
			train_data_word = train_data_word + dev_data_word

	if args.combine_train_dev_test:
		logging.info('combining training, dev data and test data')
		train_data = train_data + dev_data + test_data
		if args.model=='char_bilstm' or args.model=='char_bilstm-crf':
			train_data_word = train_data_word + dev_data_word + test_data_word

	#Check the initializer if needed
	# Declare a DynetParams object
	#dyparams = dy.DynetParams()
	# Fetch the command line arguments (optional)
	#dyparams.from_args()
	# Set some parameters manualy (see the command line arguments documentation)
	#dyparams.set_mem(2048)
	#dyparams.set_random_seed(666)
	# Initialize with the given parameters
	#dyparams.init() # or init_from_params(dyparams)

	logging.info('parsing all data')
	logging.info('train_length: %d', len(train_data))
	logging.info('dev_length: %d', len(dev_data))
	logging.info('test_length: %d', len(test_data))
	logging.info('test_length2: %d', len(test_data2))

	_, word_to_ix, tag_to_ix = get_word_label_ix(train_data,args.vocab_size)
	if args.model=='char_bilstm' or args.model=='char_bilstm-crf':
		char_to_ix, _, _ = get_word_label_ix(train_data_word,args.vocab_size)

	if len(word_to_ix)!=args.vocab_size:
		logging.info('vocab_size changed to %d',len(word_to_ix))
		args.vocab_size = len(word_to_ix)

	if args.word_embeds_file!=None and args.word_embeds_file!='None':
		logging.info('obtaining %s embeddings for words',args.word_embeds)
		word_vectors = extract_embeds(args.word_embeds_file, args.word_embed_dim, word_to_ix)
		'''
		if args.use_pretrained_embed2:
			word_vectors2 = extract_embeds(args.word_embeds_file2, args.word_embed_dim, word_to_ix)
			new_word_vectors = []
			for i,j in zip(word_vectors, word_vectors2):
				new_word_vectors.append(i+j)
			word_vectors = new_word_vectors
		'''
		#word_vectors2 = extract_embeds('utils/final_polyglot_embeds.txt', args.word_embed_dim, word_to_ix2)
		#new_word_vectors = []
		#for i,j in zip(word_vectors,word_vectors2):
		#	temp_vector = i+j
		#	new_word_vectors.append(temp_vector)

		if args.model=='char_bilstm' or args.model=='char_bilstm-crf':
			#logging.info('obtaining %s embeddings for chars',args.word_embeds)
			#char_vectors = extract_embeds(args.word_embeds_file, args.word_embed_dim, char_to_ix)
			char_vectors = None
		
	else:
		word_vectors = None
		if args.model=='char_bilstm' or args.model=='char_bilstm-crf':
			char_vectors = None

	#word_vectors = extract_embeds('../glove.twitter.27B/glove.twitter.27B.25d.txt', 25, word_to_ix)
	#word_vectors = None
	
	logging.info('Obtained word_to_ix: %d and tag_to_ix: %d ', len(word_to_ix), len(tag_to_ix))
	logging.info(tag_to_ix)

	if args.model=='char_bilstm':
		logging.info(char_to_ix)
		logging.info('using a char-level bilstm and word-level bilstm model')
		#char_vectors=None
		char_bilstm = model.ensemble_char_BiLSTMNetwork(args, char_to_ix, word_to_ix, tag_to_ix, args.num_basis, word_vectors, char_vectors)#, new_word_vectors)

		logging.info('Created the network')

		logging.info('Training the network')
		# training the network
		char_train(char_bilstm, train_data, dev_data, test_data, test_data2, train_data_word, dev_data_word, 
			test_data_word, test_data2_word, args.epochs, args.batch_size, args, tag_to_ix)

	else:
		logging.error('no such option for the model yet')
	







if __name__ == '__main__':
	main()
