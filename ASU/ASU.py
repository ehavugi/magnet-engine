import torch
import numpy as np
from scipy.signal import savgol_filter

# %% Initialize model
class ASUModel:
    def __init__(self, mdl_path, material):
        # Select GPU as default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1.Create model isntances
        # 1.Create model isntances
        self.mdl = mdl(material)
                
    
    def __call__(self, data_B, data_F, data_T, return_h_sequence=True):
        # ----------------------------------------------------------- batch execution  
        if isinstance(data_F, np.ndarray):
            return np.random.random()*10e5
        # ----------------------------------------------------------- one cycle execution 
        else:
            if return_h_sequence:
                return np.random.random()*10e5, np.linspace(0, 0, 1024)
            return np.random.random()*10e5
     
            
def mdl(material):
    return funx

def funx(B,Frequency,Temperature):
    return 1000