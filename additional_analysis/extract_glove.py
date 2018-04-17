import argparse
#import arguments

def parse_file(filename):
	data = {}
	data2={}
	with open(filename) as f:
		for i,line in enumerate(f):
			if line.strip()=='':
				pass
			else:
				word,tag = line.split('\t')
				data[word] = 1
				data2[word.lower()] = 1
	return data

def main():
	#args = arguments.get_arguments()
	train_data = parse_file('../data/oct27.train')
	dev_data = parse_file('../data/oct27.dev')
	test_data = parse_file('../data/oct27.test')
	test_data2 = parse_file('../data/daily547.test')

	whole_data = train_data.copy()
	whole_data.update(dev_data)
	whole_data.update(test_data)
	whole_data.update(test_data2)

	print ('obtained all the list of words in tweets')

	with open('../data/glove.twitter.27B/glove.twitter.27B.100d.txt','r') as f:
		for i,line in enumerate(f):
			if i%100==0:
				print (i)
			if i==0:
				pass
			else:
				word = str(line.split()[0])
				vector = [float(vec) for vec in line.split()[1:]]
				if word in whole_data:
					whole_data[word] = vector
	print ('loaded all the necessary vectors, now writing them into file')

	with open('check_final_glove_embeds.txt','w') as f:
		for key, val in whole_data.items():
			if type(val)!=type(1):
				f.write(key+' ')
				for v in val:
					f.write(str(v)+' ')
				f.write('\n')

	print ('done with it')

if __name__ == '__main__':
	main()
