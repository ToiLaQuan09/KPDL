import pickle

class SelectModel():
    def __init__(self, model, X, y, save_path) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.path = save_path

    def train_model(self):
        model = self.model.fit(self.X, self.y)  
        pickle.dump(model, open(self.path, 'wb'))


