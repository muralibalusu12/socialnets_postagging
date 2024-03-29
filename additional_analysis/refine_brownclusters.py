
from collections import Counter
import json
import numpy as np

alpha = 1.0
beta = 3.0

def obtain_wordvecs(node, final_nodes, cluster, everything, embed_dim):
	#print (' here : ', node)
	#a = input()
	if node=='*ROOT*':
		child1 = '1'
		child2 = '0'
	else:
		child1 = node+'1'
		child2 = node+'0'

	if node not in everything and (child1 in final_nodes or child2 in final_nodes or node in cluster):
		if child1 in final_nodes: 
			everything = obtain_wordvecs(child1, final_nodes, cluster, everything, embed_dim)
		if child2 in final_nodes:
			everything = obtain_wordvecs(child2, final_nodes, cluster, everything, embed_dim)

		if node in cluster:
			sum_vec = [0]*embed_dim
			for ch in cluster[node].keys():
				sum_vec += everything[ch]
			sum_vec = sum_vec/len(cluster[node])

		if child1 in final_nodes:
			if child2 in final_nodes:
				if node in cluster:
					everything[node] = (alpha*(everything[child1]+everything[child2]) + beta*sum_vec)/(alpha*2 + beta)
				else:
					everything[node] = (alpha*(everything[child1]+everything[child2]))/(alpha*2)
			else:
				if node in cluster:
					everything[node] = (alpha*(everything[child1]) + beta*sum_vec)/(alpha*1 + beta)
				else:
					everything[node] = (alpha*(everything[child1]))/(alpha*1)
		else:
			if child2 in final_nodes:
				if node in cluster:
					everything[node] = (alpha*(everything[child2]) + beta*sum_vec)/(alpha*1 + beta)
				else:
					everything[node] = (alpha*(everything[child2]))/(alpha*1)
			else:
				if node in cluster:
					everything[node] = (beta*sum_vec)/(beta)
				else:
					everything[node] = [0]*embed_dim
					
	return everything


def refine_wordvecs(node, allnodes, final_nodes, word_parents, new_everything, everything, cluster, updates, embed_dim):
	loss = 0
	if node=='*ROOT*':
		child1 = '1'
		child2 = '0'
		parent = None
	else:
		child1 = node+'1'
		child2 = node+'0'
		if node in final_nodes:
			if len(node)>1:
				parent = node[:-1]
			else:
				parent = '*ROOT*'
		else:
			parent = word_parents[node]
			child1 = None
			child2 = None

	if node not in updates and (child1 in allnodes or child2 in allnodes or node in cluster or parent in allnodes):
		if child1 in allnodes: 
			new_everything, updates, loss1 = refine_wordvecs(child1, allnodes, final_nodes, word_parents, new_everything, everything, cluster, updates, embed_dim)
			loss+=loss1
		if child2 in allnodes:
			new_everything, updates, loss2 = refine_wordvecs(child2, allnodes, final_nodes, word_parents, new_everything, everything, cluster, updates, embed_dim)
			loss+=loss2
		if node in cluster:
			for ch in cluster[node].keys():
				new_everything, updates, loss3 = refine_wordvecs(ch, allnodes, final_nodes, word_parents, new_everything, everything, cluster, updates, embed_dim)
				loss+=loss3

		dist1 = 0
		if node in cluster and node in final_nodes:
			sum_vec = [0]*embed_dim
			b1 = len(cluster[node])
			for ch in cluster[node].keys():
				sum_vec += everything[ch]
				dist1+= np.linalg.norm(everything[ch] - everything[node])
			if child1 in allnodes:
				b1+=1
				sum_vec += everything[child1]
				dist1+= np.linalg.norm(everything[child1] - everything[node])
			if child2 in allnodes:
				b1+=1
				sum_vec += everything[child2]
				dist1+= np.linalg.norm(everything[child2] - everything[node])
			if parent in allnodes:
				b1+=1
				sum_vec += everything[parent]
		else:
			b1=0
			sum_vec = [0]*embed_dim
			if child1 in allnodes:
				b1+=1
				sum_vec += everything[child1]
				dist1+= np.linalg.norm(everything[child1] - everything[node])
			if child2 in allnodes:
				b1+=1
				sum_vec += everything[child2]
				dist1+= np.linalg.norm(everything[child2] - everything[node])
			if parent in allnodes:
				b1+=1
				sum_vec += everything[parent]
		a = 10
		new_everything[node] = (a*everything[node] + (1.0/b1)*sum_vec)/(a+ (1.0/b1)*b1)

		dist2 = 0
		if node in cluster:
			for ch in cluster[node].keys():
				dist2+= np.linalg.norm(new_everything[ch] - new_everything[node])
			if child1 in final_nodes:
				dist2+= np.linalg.norm(new_everything[child1] - new_everything[node])
			if child2 in final_nodes:
				dist2+= np.linalg.norm(new_everything[child2] - new_everything[node])
		else:
			if child1 in final_nodes:
				dist2+= np.linalg.norm(new_everything[child1] - new_everything[node])
			if child2 in final_nodes:
				dist2+= np.linalg.norm(new_everything[child2] - new_everything[node])

		updates[node]=1
		loss+=(dist1-dist2)

	return new_everything, updates, loss

def extract_embeds(file_name,embed_dim):
	word_vecs = {}
	with open(file_name) as f:
		for i,line in enumerate(f):
			if len(line.split())>embed_dim:
				word = line.split()[0]
				vec = np.array([float(v) for v in line.split()[1:]])
				word_vecs[word] = vec

	#if 'unk' not in word_vecs:
	#	word_vecs['unk'] = [0]*embed_dim # adding 'unk' token to this if not there
	#word_vectors = [word_vecs[word] if word in word_vecs else word_vecs['unk'] for word,val in word_to_ix.items()]
	return word_vecs

def main():

	print ('started loading all word vectors')
	embed_dim = 100
	word_vecs = extract_embeds('../glove.twitter.27B/glove.twitter.27B.100d.txt',embed_dim)
	print ('finished loading all word vectors: ', len(word_vecs))

	cluster={}
	allnodes={}
	word_parents={}
	everything = {}
	final_nodes={}

	with open('50mpaths2','r') as f:
		for i,line in enumerate(f):
			content = line.strip().split('\t')[0]
			word = line.strip().split('\t')[1]
			if word in word_vecs:
				if content not in cluster:
					cluster[content]={}
					cluster[content][word]=word_vecs[word]
					word_parents[word] = content
				else:
					cluster[content][word]=word_vecs[word]
					word_parents[word] = content

				allnodes[content]=1
				everything[word] = cluster[content][word]
				final_nodes[content]=1

				if content[:-1] not in allnodes:
					while(1):
						content=content[:-1]
						if len(content)==0:
							break
						allnodes[content]=1

	print ('no of nodes in the tree (excluding root): ', len(allnodes))
	print ('no of words included in the tree: ', len(word_parents))

	#setting up parents and children
	parents={}
	children={}
	
	for key in sorted(allnodes.keys()):

		if len(key)!=1:
			parents[key]=key[:-1]
			final_nodes[key[:-1]]=1
		else:
			parents[key]='*ROOT*'
			final_nodes['*ROOT*']=1

		if len(key)>1 and key[:-1] not in allnodes:
			print(key)
			print('here is an issue')
			a = input()

		children[key]=[]
		if key+'1' in cluster:
			children[key].append(key+'1')
			final_nodes[key+'1']=1
		if key+'0' in cluster:
			children[key].append(key+'0')
			final_nodes[key+'0']=1

	print ('initial embeddings for nodes: ', len(everything))
	everything = obtain_wordvecs('*ROOT*', final_nodes, cluster, everything, embed_dim)
	print ('len of all nodes finally in the tree: ', len(final_nodes), len(parents), len(word_parents))
	print ('obtained embeddings for all nodes: ', len(everything))

	print ('refining all word vectors now')
	iterations = 10
	for it in range(iterations):
		node = '*ROOT*'
		updates = {}
		new_everything = {}
		new_everything, updates, loss = refine_wordvecs(node, everything, final_nodes, 
										word_parents, new_everything, everything, cluster, updates, embed_dim)
		print ('updated all the vectors: ',len(updates))
		everything = new_everything
		print ('it: ',it,' loss: ',loss)
		#a = input()

	f = open('retrofit_glove_embeds.'+str(embed_dim)+'d.txt','w')
	for word, vec in everything.items():
		if word not in final_nodes:
			f.write(word+' ')
			for v in vec:
				f.write(str(v)+' ')
			f.write('\n')


if __name__ == '__main__':
	main()