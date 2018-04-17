

def extract_embeds(file_name, embed_dim, word_to_ix):
	word_vecs = {}

	
	with open('data/glove.twitter.27B/glove.twitter.27B.100d.txt','r') as f:
		for i,line in enumerate(f):
			if len(line.split())>embed_dim:
				word = line.split()[0]
				vec = [float(v) for v in line.split()[1:]]
				if word in word_to_ix:
					word_vecs[word] = vec
	

	logging.info('no of actual glove embeddings used: '+str(len(word_vecs)))
	
	if 'unk' not in word_vecs:
		word_vecs['unk'] = [0]*embed_dim # adding 'unk' token to this if not there

	word_vectors = [word_vecs[word]if word in word_vecs 
									else word_vecs['unk']for word,val in word_to_ix.items()]
	return word_vectors

def extract_authorvecs(filename):
	author_vecs={} 
	with open(filename,'r') as f:
		for i,line in enumerate(f):
			content = line.strip().split()[1:]
			vec = [float(v) for v in content]
			author_vecs[line.strip().split()[0]]=vec
	return author_vecs

def main():
	filename = 'data/author_vecs/mention/line/mention_50_combined_pure.emb' 
	author_vecs = extract_authorvecs(filename)

if __name__ == '__main__':
	main()