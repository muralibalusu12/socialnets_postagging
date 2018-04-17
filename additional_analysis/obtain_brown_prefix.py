
def main():

	cluster={}
	with open('data/brown_clusters/50mpaths2','r') as f:
		for i,line in enumerate(f):
			contents=line.strip().split('\t')
			cluster[contents[0]]=1

	new_cluster={}
	for k,v in cluster.items():
		l = len(k)
		i=2
		while i<=l:
			new_cluster[k[:i]]=1
			i+=2
		new_cluster[k]=1

	f = open('utils/brown_prefix.txt','w')
	for k,v in new_cluster.items():
		f.write(k+'\n')
	f.close()
			

if __name__ == '__main__':
	main()