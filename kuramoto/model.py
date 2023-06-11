from . import loadDefaultParams as dp
from . import timeIntegration as ti

from neurolib.models.model import Model


class KuramotoModel(Model):
    """
    Kuramoto Model
    """

    name = "kuramoto"
    description = "Kuramoto Model"

    init_vars = ['theta_init', 'theta_ou']
    state_vars = ['theta', 'theta_ou']
    output_vars = ['theta', 'theta_ou']
    default_output = 'theta'
    input_vars = None
    default_input = None

    def __init__(self, params=None, seed=None):
        self.seed = seed
        
        integration = ti.timeIntegration
        
        if params is None:
            params = dp.loadDefaultParams(seed=self.seed)
        
        super().__init__(params=params, integration=integration)
