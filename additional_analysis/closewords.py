from sklearn.neighbors import NearestNeighbors
import numpy as np

def main():
	words={}
	filename='just_words.txt'
	with open(filename) as f:
		for i,line in enumerate(f):
			word=line.strip()
			words[word]=1


	filename='retrofit_glove_embeds.100d.txt'
	word_vecs={}
	embed_dim = 100
	with open(filename) as f:
		for i,line in enumerate(f):
			if len(line.split())>embed_dim:
				word = line.split()[0]
				vec = np.array([float(v) for v in line.split()[1:]])
				if word in words:
					word_vecs[word]=vec
	print('done loading vectors: ',len(word_vecs))

	words=[]
	wordvecs=[]
	for k,v in word_vecs.items():
		words.append(k)
		wordvecs.append(v)

	print('starting nearest neighbors')
	nbrs = NearestNeighbors(n_neighbors=10).fit(wordvecs)
	distances, indices = nbrs.kneighbors(wordvecs)
	print('finished nearest neighbors ')

	close_words=[]
	for ind in indices:
		close_word=[]
		for i in ind:
			close_word.append(words[i])
			close_words.append(close_word)

	print('converted to words and writing to file')

	with open('close_words.txt','w') as f:
		for closeword in close_words:
			for i in closeword:
				f.write(i+' ')
			f.write('\n')

if __name__ == '__main__':
	main()