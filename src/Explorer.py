import pandas as pd
import matplotlib.pyplot as plt

class Explorer:
	def __init__(self, dataset):
		self.dataset = dataset

	def plot_histograms(self):
		num_features = self.dataset.params["numerical_features"]
		plt.tight_layout()
		plt.show()

	def make_table(self):
		stats = pd.DataFrame()
		stats["Mean"] = self.dataset.mean()
		stats["Std"] = self.dataset.std()
		stats["Missing Rate"] = self.dataset.isnull().mean()
		pd.display(stats)
		
	def explore(self):
		self.plot_histograms()
		self.make_table()
	