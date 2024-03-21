import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from numpy import log10
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from os.path import isfile
from typing import List, Dict
from torch.utils.data import DataLoader


from main_net import Net, load_DATASET,loadModel
from utils import dict_to_json,json_to_dict
from losses import *

def _get_psnr_list(image_array1: np.ndarray, image_array2: np.ndarray) -> List[float]:
    """_get_psnr_list
    compute Peak Signal to Noise Ratios for each 2 elements of the input arrays
    
    # TODO: find faster, more general solution / implementation

    Args:
        image_array1 (np.ndarray): np array of 2d images
        image_array2 (np.ndarray): np array of 2d images

    Returns:
        List[float]: psnr value for each element-pair in the input array.
    """
    psnr_list = []
    for im1, im2 in zip(image_array1, image_array2):
        try:
            psnr_value = psnr(im1, im2)
        except Exception:
            data_range = max(im1.max() - im1.min(), im2.max() - im2.min()) # refer to psnr function documentation.
            psnr_value = psnr(im1, im2, data_range=data_range)
        
        psnr_list.append(psnr_value)

    return psnr_list

class ModelEvaluation:

    def __init__(self, model_name: str, data_name : str, rot180: bool = False, circ_shift: bool = False, test_loader: DataLoader = None):
        """
        model evaluation class - compute performance metric for test dataset and save visual examples.

        Args:
            model_name (str): model / architecture / version name
            data_name (str): dataset name
            rot180 (bool, optional): enable rotations. Defaults to False.
            circ_shift (bool, optional): enable circular shift. Defaults to False.
            test_loader (DataLoader, optional): dataloader. Defaults to None.
        """
        # load model
        model = loadModel(model_name,device='cpu',data_name=data_name)
        model.eval()

        # load dataset    
        if(test_loader == None):
            test_loader = load_DATASET(data_name, "test",resize_len=32,batch_size = 32, random_transform_enabled = False)
        
        test_fft = test_loader.dataset.data
        self._test_target = test_loader.dataset.targets

        # prediction for test dataset
        self._model_prediction = model(test_fft)
        
        # perform inverse normalization for model prediction and ground truth
        std = 0.3081
        mean = 0.1307
        std_inv = 1/std
        mean_inv = -mean*std_inv
        inv_normalization = transforms.Normalize(mean_inv,std_inv)
        self._test_target = inv_normalization(self._test_target)
        self._model_prediction = inv_normalization(self._model_prediction)
            
        # create output folder for results
        self._results_folderpath = Path.cwd().joinpath("out", data_name, model_name)
        self._results_folderpath.mkdir(parents=True, exist_ok=True)

        # additional params
        self.rot180 = rot180
        self.circ_shift = circ_shift 
        
        
    def performance_metrics(self, overwrite: bool = False) -> Dict[str, float]:
        """
        compute performance metrics as described in the paper

        Args:
            overwrite (bool, optional): overwrite existing results. Defaults to False.

        Returns:
            Dict[str, float]: dict of metric values
        """
        if((not overwrite) and isfile(self._results_folderpath.joinpath("metrics.json"))):
            print(f'Model Metrics were already computed, use overwrite = True or delete {self._results_folderpath.joinpath("metrics.json")}')
            return json_to_dict(self._results_folderpath.joinpath("metrics.json"))
        
        self._aligend_model_prediction = AlignPrediction(self._model_prediction,self._test_target,self.rot180,self.circ_shift)
        
        # compute Mean Squared Error (MSE) and Mean Absolute Error (MAE)
        mse = torch.mean((self._test_target - self._aligend_model_prediction)**2).item()
        mae = torch.mean(torch.abs(self._test_target - self._aligend_model_prediction)).item()

        # convert tensors to numpy arrays
        gt_np = self._test_target.numpy() # type: ignore
        pred_np = self._aligend_model_prediction.detach().numpy()

        # compute mean Structural Similarity Index (SSIM)
        ssim_value = ssim(gt_np, pred_np, multichannel=True)

        # compute mean Peak Signal-to-Noise Ratio (PSNR)
        psnr_value = 20*log10(1/mse)

        metrics_dict = {"mse": mse, "mae": mae, "ssim": ssim_value, "psnr": psnr_value}
        
        dict_to_json(metrics_dict, self._results_folderpath.joinpath("metrics.json"))
        
        return metrics_dict

    def save_example_images(self, n_test_images: int = 5):
        """
        save visual examples of model prediction and ground truth.

        Args:
            n_test_images (int, optional): number of examples to save. Defaults to 5.
        """
        # create dir
        images_folderpath = self._results_folderpath.joinpath("images")
        images_folderpath.mkdir(exist_ok=True)
        
        # save visual examples
        for image_ind in range(1, n_test_images + 1):
            plt.imshow(self._test_target[image_ind]) # type: ignore
            plt.axis('off')
            plt.savefig(images_folderpath.joinpath(f"{image_ind}-ground-truth.png")) # FIXME: doesn't save the fig for some reason...

            plt.imshow(self._model_prediction[image_ind].detach())
            plt.axis('off')
            plt.savefig(images_folderpath.joinpath(f"{image_ind}-prediction.png"))

# if __name__ == "__main__":
#     me = ModelEvaluation(model_name = "ConvOnly_model", data_name= "MNIST")
#     me.performance_metrics(overwrite=True)
#     me.save_example_images()
