from src.Dataset import Dataset
from src.Explorer import Explorer
from src.Trainer import Trainer

class Pipeline:
    def __init__(self, params, data):
        self.params = params
        self.data = data
        self.random_state = params["random_state"]

    def preprocess(self):
        self.dataset = Dataset(self.params["Dataset"], self.data, self.random_state)
        self.dataset.prepare_data()
    
    def explore_data(self):
        self.explorer = Explorer(self.dataset)
        self.explorer.explore()
    
    def train(self):
        self.trainer = Trainer(self.params["Trainer"], self.dataset)
        self.trainer.train()
        results, results_hpo = self.trainer.get_results()
        print(results)
        print(results_hpo)
    
    def run(self):
        self.preprocess()
       # self.explore_data()
        self.train()
        