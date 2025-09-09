import matplotlib
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import logging
from sklearn.preprocessing import MinMaxScaler
from get_hsi import get_data
from lib.ssn.ssn import get_number_of_superpixels
from lib.utils.estimate_num_superpixels import estimate_superpixel_count
from subspace_clustering import subspace_clustering
from datetime import datetime
from calculate_distance import Distance
from visualize import plot_heatmap
import pytorch_optimizer as ptop
import numpy as np
import torch
import torch.optim as optim
from lib.utils.meter import Meter
from model import SSNModel
from lib.utils.loss import reconstruct_loss_with_mse, spectral_compact, \
    soft_connectivity_loss,  compute_sparsity_metrics, \
    compute_smooth_nonzero_loss
import random
import torch.nn.functional as F
torch.use_deterministic_algorithms(True)


# Configure logging
def set_log(log_dir):
    current_time = datetime.now().strftime("%Y_%m_%d|%H_%M_%S")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/{current_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def sklearn_normalize_per_channel(img):
    h, w, c = img.shape
    reshaped = img.reshape(-1, c)  # shape: (h*w, c)

    scaler = MinMaxScaler(feature_range=(0, 255))
    norm = scaler.fit_transform(reshaped)
    return norm.reshape(h, w, c).astype(np.float32)

from PIL import Image
import torchvision.transforms as transforms

def log_matrix_to_tensorboard(writer, matrix, tag, step):
    buf = plot_heatmap(matrix, title=tag)
    image = Image.open(buf)
    transform = transforms.ToTensor()
    image_tensor = transform(image)  # shape: [3, H, W]
    writer.add_image(tag, image_tensor, global_step=step)

target = None

def update_param(args, data, label, model, optimizer, scheduler, device, iter, iters, writer=None, logger=None, early_stop=False, target=None):
    inputs = data.to(device)
    height, width = inputs.shape[-2:]

    coords = torch.stack(torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij"
    ), 0).float()
    Q, H, feat, superpixel_recon_norm, C, Z, superpixel_norm, affinity_matrix = model(Feature=inputs, coordinate=coords)

    recons_loss = torch.sum(torch.square(superpixel_recon_norm - superpixel_norm))

    compact_loss = reconstruct_loss_with_mse(Q, coords.reshape(coords.shape[0], -1), H)
    spectralcompact = spectral_compact(Q, inputs.reshape(inputs.shape[0], -1)+model.noise.reshape(inputs.shape[0], -1), H)



    Q_reshape = Q.reshape([Q.shape[0], height, width])
    connect_loss = soft_connectivity_loss(Q_reshape)

    non_zero_ratio, entropy1 = compute_sparsity_metrics(C)
    print(non_zero_ratio, entropy1)

    sparss_ = compute_smooth_nonzero_loss(Z)
    print(sparss_)
    sparse_loss = torch.norm(C, p=1, dim=0).mean()

    loss_noise = torch.mean(torch.square(model.noise))

    super_pixel_loss = 1.0*spectralcompact + connect_loss
    representation_loss = 2.0*recons_loss + 1.0*(sparse_loss + entropy1)

    loss = args.weight_representation*representation_loss + 1.0*super_pixel_loss + args.weight_noise*loss_noise

    if (iter % 50 == 0 or iter == iters - 1 or early_stop==True) and label is not None:

        acc, kappa, nmi = subspace_clustering(C, label, H, iter, logger)
        if writer is not None:
            writer.add_scalar("Metric/Accuracy", acc, iter)
            writer.add_scalar("Metric/Kappa", kappa, iter)
            writer.add_scalar("Metric/NMI", nmi, iter)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step(loss)

    if writer is not None:
        writer.add_scalar("Loss/Total", loss.item(), iter)
        writer.add_scalar("Loss/Compact", compact_loss.item(), iter)
        writer.add_scalar("Loss/Spectral", spectralcompact.item(), iter)
        writer.add_scalar("Loss/Reconstruction", recons_loss.item(), iter)
        writer.add_scalar("Loss/Sparse", sparse_loss.item(), iter),
        writer.add_scalar("Loss/super_pixel", super_pixel_loss.item(), iter)
        writer.add_scalar("Loss/clustering", representation_loss.item(), iter)
        writer.add_scalar("Loss/entropy1", entropy1.item(), iter)
        writer.add_scalar("Loss/connect_loss", connect_loss.item(), iter)
        writer.add_scalar("Loss/loss_noise", loss_noise.item(), iter)


    return {
        "loss": loss.item(),
        "compact": compact_loss.item(),
        "spectral": spectralcompact.item(),
        "recons_loss:": recons_loss.item(),
        "sparse_loss:": sparse_loss.item(),
    }, target




def train(cfg):
    device = args.device
    from torch.utils.tensorboard import SummaryWriter
    if cfg.dataset == "salinas":
        data, label = get_data(img_path="./data/salinas/Salinas_corrected.mat",
                               label_path="./data/salinas/Salinas_gt.mat")
        data = sklearn_normalize_per_channel(data)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')  # e.g., 20250411-153045
        log_dir = f'./salinas/{timestamp}'
        writer = SummaryWriter(log_dir=log_dir)

    elif cfg.dataset == "trento":
        data, label = get_data(img_path="./data/trento/Trento.mat",
                               label_path="./data/trento/Trento_gt.mat")
        data = sklearn_normalize_per_channel(data)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')  # e.g., 20250411-153045
        log_dir = f'./trento/{timestamp}'
        writer = SummaryWriter(log_dir=log_dir)
    elif cfg.dataset == "urban":
        data, label = get_data(img_path="./data/urban/Urban_corrected.mat",
                               label_path="./data/urban/Urban_gt.mat")
        data = sklearn_normalize_per_channel(data)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')  # e.g., 20250411-153045
        log_dir = f'./urban/{timestamp}'
        writer = SummaryWriter(log_dir=log_dir)

    if cfg.dataset == "salinas":
        cfg.nspix = 1642
    elif cfg.dataset == "trento":
        cfg.nspix = 1007
    elif cfg.dataset == "urban":
        cfg.nspix = 697
    else:
        numer_pixel = estimate_superpixel_count(data)
        logger.info("number of pixels: %d", numer_pixel)
        cfg.nspix = numer_pixel

    label_index, label_count = np.unique(label, return_counts=True)
    print("label_index",label_index, "label_count",label_count)

    data = np.expand_dims(data, axis=0)
    data = np.swapaxes(data, 1, 3)
    data = torch.tensor(data)
    import kornia.filters as kf
    data = kf.gaussian_blur2d(data, kernel_size=(7, 7), sigma=(2.5, 2.5), border_type='reflect')
    data = torch.squeeze(data)

    numer_superpixel, num_spixels_width, num_spixels_height, init_spix_indices = get_number_of_superpixels(data, cfg.nspix)

    distance = Distance(init_spix_indices=init_spix_indices, num_spixels_w=num_spixels_width, num_spixels_h=num_spixels_height, device=device)
    distance.innitialize()
    model = SSNModel(numer_superpixel, labels=label, data = data, n_iter=cfg.niter, origin_nspixle=cfg.nspix, cal = distance, device=device).to(device)
    lamda_param = []
    other_param = []

    for name, param in model.named_parameters():
        if 'lamda' in name:
            lamda_param.append(param)
        else:
            other_param.append(param)


    optimizer = optim.Adam([
        {'params': other_param, 'lr': cfg.lr},  
        {'params': lamda_param, 'lr': cfg.lr*0.01}  
    ])


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        threshold=1e-4,
        threshold_mode='rel',
        min_lr=1e-8
    )


    meter = Meter()
    patience = 500
    min_delta = 0.0001
    best_loss = float('inf')
    no_improve_epochs = 0
    early_stop = False
    target = None
    iterations = 0
    while iterations < cfg.train_iter:

        metric, target = update_param(args,data, label, model, optimizer, scheduler, device, iterations, cfg.train_iter, writer=writer, logger=logger, early_stop = early_stop, target = target)
        if early_stop==True:
            break
        meter.add(metric)
        state = meter.state(f"[{iterations}/{cfg.train_iter}]")
        logger.info(state)
        current_loss = meter.params["loss"]
        logger.info(f"[Iter {iterations}] LR: {optimizer.param_groups[0]['lr']:.8f}")
        print("model.lamda: ", model.lamda)
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            no_improve_epochs = 0
            logger.info(f"Saving best model at iteration {iterations}")
        else:
            logger.info(f"No improvement: current loss {current_loss:.6f}, best loss {best_loss:.6f}")
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                early_stop = True
                logger.warning(f"Early stopping at iteration {iterations}!")
        iterations += 1

    logger.info("Training finished. Best model loaded.")


if __name__ == "__main__":
    set_seed(55)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./log", type=str, help="/path/to/output directory")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=300, type=int)
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=300, type=int, help="number of superpixels")
    parser.add_argument("--weight", default="weight", type=str, help="folder where the weight will be saved")
    parser.add_argument("--weight_representation", default="250", type=float, help="folder where the weight will be saved")
    parser.add_argument("--weight_noise", default="50", type=float, help="folder where the weight will be saved")
    parser.add_argument("--dataset", default="trento", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--log", default="urban_test", type=str)
    args = parser.parse_args()

    if args.weight_representation>0:
        args.lr = 0.1


    logger = set_log(args.log)

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("========== Training Configuration ==========")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("============================================")
    train(args)
