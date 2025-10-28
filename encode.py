# encodes a directory of images using ImageGS

from types import SimpleNamespace

import torch
from pathlib import Path
import yaml
import fire
# from model_light import GaussianSplatting2D
from model_light_improved import GaussianSplatting2D, GSImage, BatchedGSImage
from utils.misc_utils import load_cfg
from PIL import Image
import json
import tqdm
import sys

from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset


def get_gaussian_cfg(args):
    gaussian_cfg = f"num-{args.num_gaussians:d}"
    if args.disable_inverse_scale:
        gaussian_cfg += f"_scale-{args.init_scale:.1f}"
    else:
        gaussian_cfg += f"_inv-scale-{args.init_scale:.1f}"
    if not args.quantize:
        args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits = 32, 32, 32, 32
    min_bits = min(args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits)
    max_bits = max(args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits)
    if min_bits < 4 or max_bits > 32:
        raise ValueError(
            f"Bit precision must be between 4 and 32 but got: {args.pos_bits:d}, {args.scale_bits:d}, {args.rot_bits:d}, {args.feat_bits:d}")
    gaussian_cfg += f"_bits-{args.pos_bits:d}-{args.scale_bits:d}-{args.rot_bits:d}-{args.feat_bits:d}"
    if not args.disable_topk_norm:
        gaussian_cfg += f"_top-{args.topk:d}"
    gaussian_cfg += f"_{args.init_mode[0]}-{args.init_random_ratio:.1f}"
    return gaussian_cfg


def get_log_dir(args):
    gaussian_cfg = get_gaussian_cfg(args)
    loss_cfg = f"l1-{args.l1_loss_ratio:.1f}_l2-{args.l2_loss_ratio:.1f}_ssim-{args.ssim_loss_ratio:.1f}"
    folder = f"{gaussian_cfg}_{loss_cfg}"
    if args.downsample:
        folder += f"_ds-{args.downsample_ratio:.1f}"
    if not args.disable_lr_schedule:
        folder += f"_decay-{args.max_decay_times:d}-{args.decay_ratio:.1f}"
    if not args.disable_prog_optim:
        folder += "_prog"
    return f"{args.log_root}/{args.exp_name}/{folder}"


# python main.py --input_path="media/images/anime-1_2k.png" --exp_name="results/test/anime-1_2k" --num_gaussians=10000 --quantize

def main(args):
    args.log_dir = get_log_dir(args)
    ImageGS = GaussianSplatting2D(args)

    # loaded images are (C, H, W) in [0, 1] range
    if args.eval:
        ImageGS.render(render_height=args.render_height)
    else:
        ImageGS.optimize()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None, start_idx=0, end_idx=-1):
        self.image_dir = Path(image_dir)
        
        # Get all image files with common extensions
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            self.image_paths.extend(list(self.image_dir.glob(ext.upper())))

        self.start_idx, self.end_idx = start_idx, end_idx
        self.image_paths = self.image_paths[self.start_idx:self.end_idx]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return only the image (no label)
        return image, [str(image_path)]



def encode_directory(path="../miniimagenet_256/",
                     start_idx=0,
                     end_idx=-1,
                     config="cfgs/default_improved.yaml",
                     outdir="miniimagenet_256_out_batched",
                     num_gaussians=4000,
                     min_steps=700,
                     max_steps=5000,
                     target_psnr=32,
                     imsize=256,
                     batsize=2,
                     numworkers=4,
                     save=True,
                     overwrite=False,
                     ):
    
    localargs = locals().copy()
    outpath = Path(outdir) / f"num-{num_gaussians}_psnr-{target_psnr}"
    outpath.mkdir(parents=True, exist_ok=True)

    gs_dir = outpath / "gs"
    recons_dir = outpath / "recons"
    diff_dir = outpath / "diff"
    gs_dir.mkdir(parents=True, exist_ok=True)
    recons_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load default config
    with open(config, "r", encoding='utf-8') as file:
        cfg: dict = yaml.safe_load(file)
        cfg["min_steps"] = min_steps
        cfg["max_steps"] = max_steps
        cfg = SimpleNamespace(**cfg)

    with open(outpath / "args.yaml", "w", encoding='utf-8') as f:
        localargs["cfg"] = cfg.__dict__
        yaml.dump(localargs, f)

    # load data
    transform = transforms.Compose([# transforms.Resize(imsize),
                                    # transforms.CenterCrop(imsize),
                                    transforms.ToTensor()])
    dataset = ImageDataset(image_dir=path, transform=transform, start_idx=start_idx, end_idx=end_idx)
    print(f"Number of examples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batsize, shuffle=True, num_workers=numworkers)

    # initialize ImageGS
    gs = GaussianSplatting2D(cfg)

    print("initialized")
    plot_images = False
    test_reload = True

    for batch in tqdm.tqdm(dataloader):
        images, paths = batch      # (B, C, H, W), in range (0, 1)
        images = images.to(device)
        impath = Path(paths[0][0])
        if not overwrite and (gs_dir / f"{impath.stem}.pt").exists():
            # print("image exists")
            continue
        gsimage, recons_image, metrics = gs.optimize(images, total_num_gaussians=num_gaussians, target_psnr=target_psnr)
        if save:
            torch.save(gsimage.state_dict(), gs_dir / f"{impath.stem}.pt")
            to_pil_image(recons_image.clamp_(0, 1)).save(recons_dir / f"{impath.name}")
            to_pil_image((images[0] - recons_image).abs().clamp_(0, 1)).save(diff_dir / f"diff_{impath.name}")

            if test_reload:
                test_reload = False
                gsimage = GSImage.from_statedict(torch.load(gs_dir / f"{impath.stem}.pt", map_location=device))
                reload_recons = gs.forward(gsimage)[0]
                diff = (reload_recons - recons_image).abs()
                print("reloaded diff max and mean:", diff.max().item(), diff.mean().item())


        print(metrics)
        if plot_images:
            to_pil_image(recons_image.clamp_(0, 1)).save("recons.png"); to_pil_image(images[0].clamp_(0, 1)).save("original.png"), to_pil_image((images[0] - recons_image).abs().clamp_(0, 1)).save("diff.png")

    print("done iterating")


class GSDataset(Dataset):
    def __init__(self, path="miniimagenet_256_out/num-4000_psnr-32/gs", num_examples=-1):
        self.path = Path(path)
        self.files = list(self.path.glob("*.pt"))
        print(f"Found {len(self.files)} files in {self.path}")
        if num_examples > 0:
            self.files = self.files[:num_examples]
        print(f"Using {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        gsimage = GSImage.from_statedict(torch.load(file_path))
        return gsimage
    
    @staticmethod
    def collate_fn(batch):
        gsimages = batch
        batchedgsimage = BatchedGSImage.from_gsimages(gsimages)
        batch = batchedgsimage
        return batch
    

def load_gs(
        path="miniimagenet_256_out/num-4000_psnr-32/gs",
        batch_size=20,
        num_workers=4,
        num_examples=-1,
        gsconfig="cfgs/default_improved.yaml",
    ):
    ds = GSDataset(path=path, num_examples=num_examples)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=GSDataset.collate_fn)
    
    with open(gsconfig, "r", encoding='utf-8') as file:
        cfg: dict = yaml.safe_load(file)
        cfg["min_steps"] = 5000
        cfg["max_steps"] = 10000
        cfg = SimpleNamespace(**cfg)

    imagegs = GaussianSplatting2D(cfg)

    xy_stats = [0, 0, 0, 0]
    scale_stats = [0, 0, 0, 0]
    rot_stats = [0, 0]
    feat_stats = [0, 0, 0, 0, 0, 0]

    # (inverse) scale should be between 0.1 (pretty large and clipped) and 20 (invisible at 256 res)
    # we shouldn't allow too large scales
    # rotation can be between 0 and pi
    # xy 

    for batch in tqdm.tqdm(dl):
        gs = batch
        gs.rot.data = gs.rot.data % torch.pi
        gs.scale.data = gs.scale.data.abs()
        xy_stats = [min(xy_stats[0], gs.xy[..., 0].min().item()), 
                    max(xy_stats[1], gs.xy[..., 0].max().item()),
                    min(xy_stats[2], gs.xy[..., 1].min().item()), 
                    max(xy_stats[3], gs.xy[..., 1].max().item()),
                    ]
        scale_stats = [min(scale_stats[0], (1/gs.scale[..., 0]).min().item()),
                       max(scale_stats[1], (1/gs.scale[..., 0]).max().item()),
                       min(scale_stats[2], (1/gs.scale[..., 1]).min().item()),
                       max(scale_stats[3], (1/gs.scale[..., 1]).max().item()),
                       ]
        rot_stats = [min(rot_stats[0], gs.rot.min().item()), 
                     max(rot_stats[1], gs.rot.max().item())]
        
        feat_stats = [min(feat_stats[0], gs.feat[..., 0].min().item()),
                      max(feat_stats[1], gs.feat[..., 0].max().item()),
                      min(feat_stats[2], gs.feat[..., 1].min().item()),
                      max(feat_stats[3], gs.feat[..., 1].max().item()),
                      min(feat_stats[4], gs.feat[..., 2].min().item()),
                      max(feat_stats[5], gs.feat[..., 2].max().item()),
                      ]
        x = gs.pack()
        # print(gs)

    print("xy stats:", xy_stats)
    print("scale stats:", scale_stats)
    print("rot stats:", rot_stats)  
    print("feat stats:", feat_stats)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        load_gs()
    else:
        fire.Fire({
            "encode_dir": encode_directory,
            "load_gs": load_gs,
        })
    # fire.Fire(encode_directory)
    # torch.hub.set_dir("models/torch")
    # parser = argparse.ArgumentParser()
    # parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    # arguments = parser.parse_args()
    # arguments.input_path = "media/images/anime-1_2k.png"
    # arguments.exp_name = "dirout"
    # Path(arguments.exp_name).mkdir(parents=True, exist_ok=True)
    # main(arguments)
