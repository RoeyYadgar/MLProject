import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from pathlib import Path

from main_net import Net, load_MNIST
from utils import dict_to_json


def test_model(model_name: str):
    # load model
    model = torch.load('model_quarter_translation_cross_correlation.pt')
    model = model.to('cpu')
    model.eval()

    # intitalize test dataset
    test_loader = load_MNIST('test', model.image_size, random_transform_enabled=False)
    test_fft = test_loader.dataset.data # type: ignore
    test_target = test_loader.dataset.targets # type: ignore

    # prediction for test dataset
    model_prediction = model(test_fft)

    # compute Mean Squared Error (MSE), Mean Absolute Error (MAE)
    mse = torch.mean((test_target - model_prediction)**2).item()
    mae = torch.mean(torch.abs(test_target - model_prediction)).item()

    # Compute Structural Similarity Index (SSIM)
    # Convert tensors to numpy arrays
    gt_np = test_target.numpy() # type: ignore
    pred_np = model_prediction.detach().numpy()
    ssim_value = ssim(gt_np, pred_np, multichannel=True)  # multichannel=True for RGB images

    # Compute Peak Signal-to-Noise Ratio (PSNR)
    # psnr_value = psnr(gt_np, pred_np)

    # create output folder for results
    results_folderpath = Path.cwd().joinpath("out", "testing", model_name)
    results_folderpath.mkdir(parents=True, exist_ok=True)

    metrics_dict = {"mse": mse, "mae": mae, "ssim": ssim_value}
    
    print(metrics_dict)
    dict_to_json(metrics_dict, results_folderpath.joinpath("metrics.json"))

    # save visual example
    image_ind = 1
    plt.imshow(test_target[image_ind]) # type: ignore
    plt.savefig(results_folderpath.joinpath("test_out.png")) # FIXME: doesn't save the fig for some reason...

    plt.imshow(model_prediction[image_ind].detach())
    plt.savefig(results_folderpath.joinpath("prediction_out.png"))

if __name__ == "__main__":
    test_model(model_name = "model_quarter_translation_cross_correlation")
