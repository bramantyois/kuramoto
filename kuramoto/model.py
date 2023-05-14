from . import loadDefaultParams as dp
from . import timeIntegration as ti

from neurolib.models.model import Model


class KuramotoModel(Model):
    def __init__(self, params=None, seed=None):
        self.seed = seed
        
        integration = ti.timeIntegration
        
        if params is None:
            params = dp.loadDefaultParams(seed=self.seed)
        
        super().__init__(params=params, integration=integration)