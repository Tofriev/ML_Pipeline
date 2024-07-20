from src.Dataset import Dataset
from src.Explorer import Explorer
from src.Trainer import Trainer
import csv

class Pipeline:
    def __init__(self, params, data, frac_ids=None):
        self.params = params
        self.data = data
        self.random_state = params["random_state"]
        self.dataset = Dataset(self.params["Dataset"], self.data, self.random_state, frac_ids)

    def preprocess(self):
        self.dataset.prepare_data()
    
    def explore_data(self):
        self.explorer = Explorer(self.dataset)
        self.explorer.explore()
    
    def train(self):
        self.trainer = Trainer(self.params["Trainer"], self.dataset)
        self.trainer.train()
        self.results, self.results_hpo, self.feature_importances = self.trainer.get_results()
        self.trainer.plot_feature_importances()
        self.trainer.plot_ebm_feature_contributions()
        print(self.results)
        print(self.results_hpo)
    
    def run(self):
        self.preprocess()
        # self.explore_data()
        self.train()


    def return_results(self, with_hpo=False):
        if with_hpo:
            return self.results, self.results_hpo
        else:
            return self.results, self.feature_importances
    def return_frac_ids(self):
        ids = self.dataset.make_fractional_ids()
        print(len(ids))
        return ids
        