import numpy as np
import struct
import matplotlib.pyplot as plt
import tqdm

class mnist:
	def __init__(self,trl=60000,tkl=10000):
		print()
		self.dir = "/home/codog/ML/dataset/MNIST/"
		print("MNIST DIR:"+self.dir)
		self.train_len = trl
		self.test_len = tkl
		##--------------##
		self.train_images = []
		with open(self.dir+"train-images.idx3-ubyte","rb") as f:
			f.seek(16)
			print("Reading  train-images:")
			for _ in tqdm.tqdm(range((self.train_len))):
				tmp = []
				for i in range(28):
					for j in range(28):
						tmp.append(struct.unpack('B',f.read(1))[0])
				self.train_images.append( np.array(tmp).reshape(28,28).astype('int') )
		##--------------##
		self.train_images = np.array(self.train_images)
		##--------------##
		self.test_images = []
		with open(self.dir+"t10k-images.idx3-ubyte","rb") as f:
			f.seek(16)
			print("Reading  test-images:")
			for _ in tqdm.tqdm(range((self.test_len))):
				tmp = []
				for i in range(28):
					for j in range(28):
						tmp.append(struct.unpack('B',f.read(1))[0])
				self.test_images.append( np.array(tmp).reshape(28,28).astype('int') )
		##--------------##
		self.test_images  = np.array(self.test_images)
		##--------------##
		self.train_labels = []
		with open(self.dir+"train-labels.idx1-ubyte","rb") as f:
			f.seek(8)
			print("Reading  train-labels:")
			for _ in tqdm.tqdm(range((self.train_len))):
				self.train_labels.append( struct.unpack('B',f.read(1))[0] )
		##--------------##
		self.train_labels = np.array(self.train_labels)
		##--------------##
		self.test_labels = []
		with open(self.dir+"t10k-labels.idx1-ubyte","rb") as f:
			f.seek(8)
			print("Reading test-labels:")
			for _ in tqdm.tqdm(range((self.test_len))):
				self.test_labels.append( struct.unpack('B',f.read(1))[0] )
		##--------------##
		self.test_labels = np.array(self.test_labels)
		##--------------##
		self.doc = ["train_images","test_images","train_labels","test_labels"]

	def show(self,img,title="MNIST"):
		plt.imshow(img)
		plt.title(title)
		plt.show()

## a = mnist()