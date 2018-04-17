import argparse
#import arguments

def parse_file(filename):
	data = {}
	with open(filename) as f:
		for i,line in enumerate(f):
			if line.strip()=='':
				pass
			else:
				word,tag = line.split('\t')
				data[word] = 1
	return data

def main():
	#args = arguments.get_arguments()
	train_data = parse_file('../data/oct27.train')
	dev_data = parse_file('../data/oct27.dev')
	test_data = parse_file('../data/oct27.test')

	whole_data = train_data.copy()
	whole_data.update(dev_data)
	whole_data.update(test_data)

	print ('obtained all the list of words in tweets')

	with open('../../word2vec_vectors/GoogleNews-vectors-negative300.txt','r') as f:
		for i,line in enumerate(f):
			if i%100==0:
				print (i)
			if i==0:
				pass
			else:
				word = str(line.split()[0])
				#print (word)
				vector = [float(vec) for vec in line.split()[1:]]
				if word in whole_data:
					whole_data[word] = vector
	print ('loaded all the necessary vectors, now writing them into file')

	with open('final_word2vec_embeds.txt','w') as f:
		for key, val in whole_data.items():
			if type(val)!=type(1):
				f.write(key+' ')
				for v in val:
					f.write(str(v)+' ')
				f.write('\n')

	print ('done with it')

if __name__ == '__main__':
	main()
