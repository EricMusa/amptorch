

class Dataset():

    def get_trajectory(self):
        pass


def new_model_configs(**kwargs):
    pass


def new_model(config, model_type):
    pass


def train_model(model, retrain=True):
    pass


class NNModel():
    
    def train(self, training_data):
        pass

    def predict(self, point):
        pass
    

class EnsembleModel(NNModel):
    pass

class DropoutModel(NNModel):
    pass


class LSKDEModel(NNModel):
    
    def pca(self, point):
        pass

    def kde(self, point):
        pass

    def uncertainty(self, point):
        pass


class DensityModel(NNModel):

    def point_density(self, point):
        pass

    def dataset_density(self):
        pass

    def uncertainty(self, point):
        pass






