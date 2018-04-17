import argparse
#import arguments

def parse_file(filename):
	data = {}
	wl_data={}
	with open(filename) as f:
		for i,line in enumerate(f):
			if line.strip()=='':
				pass
			else:
				word,tag = line.split('\t')
				data[word.strip()] = 1
				wl_data[word.strip().lower()] = 1
	return data, wl_data

def main():
	#args = arguments.get_arguments()
	train_data, wl_train_data = parse_file('../data/oct27.train')
	dev_data, wl_dev_data = parse_file('../data/oct27.dev')
	test_data, wl_test_data = parse_file('../data/oct27.test')
	test_data2, wl_test_data2 = parse_file('../data/daily547.test')

	whole_data = train_data.copy()
	whole_data.update(dev_data)
	whole_data.update(test_data)
	whole_data.update(test_data2)

	wl_whole_data = wl_train_data.copy()
	wl_whole_data.update(wl_dev_data)
	wl_whole_data.update(wl_test_data)
	wl_whole_data.update(wl_test_data2)

	print ('obtained all the list of words in tweets')

	with open('all_words.txt','w') as f:
		for key, val in whole_data.items():
			f.write(key)
			f.write('\n')

	with open('all_words_lower.txt','w') as f:
		for key, val in wl_whole_data.items():
			f.write(key)
			f.write('\n')

	wl_whole_data.update(whole_data)
	with open('combined_words.txt','w') as f:
		for key, val in wl_whole_data.items():
			f.write(key)
			f.write('\n')

	print ('done with it')

if __name__ == '__main__':
	main()
