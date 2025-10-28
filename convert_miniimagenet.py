import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from model_light_improved import GaussianSplatting2D, GSImage, BatchedGSImage
from pathlib import Path
import tqdm

import fire
import sys
import yaml
from types import SimpleNamespace
import numpy as np


class ImageNetDataset(Dataset):
    def __init__(self, rootdir, transform=None, return_paths=False, from_idx=0, to_idx=-1, max_size=-1):
        super(ImageNetDataset, self).__init__()
        self.imagepaths = []
        self.classes = {}
        for path in tqdm.tqdm(Path(rootdir).rglob('**/*')):
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.imagepaths.append(path)
                class_name = path.stem.split("_")[0]
                if class_name not in self.classes:
                    self.classes[class_name] = len(self.classes)
                if max_size > 0 and len(self.imagepaths) >= max_size:
                    break

        self.imagepaths = self.imagepaths[from_idx:to_idx if to_idx > 0 else None]
        print("Found {} images belonging to {} classes.".format(len(self.imagepaths), len(self.classes)))

        # process classes
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        path = self.imagepaths[idx]
        class_id = self.classes[path.stem.split("_")[0]]
        image = read_image(str(path)).float() / 255.0  # Normalize to [0, 1]
        if self.transform:
            image = self.transform(image)
        if self.return_paths:
            return image, torch.Tensor([class_id]).to(torch.long).to(image.device), str(path)
        else:
            return image, torch.Tensor([class_id]).to(torch.long).to(image.device)
    

class ImageNetGSDataset(Dataset):
    def __init__(self, path, transform=None, return_paths=False):
        super().__init__()
        self.gspaths = []
        self.classes = {}
        for path in tqdm.tqdm(Path(path).rglob('*')):
            if path.suffix.lower() in [".pt"]:
                self.gspaths.append(path)
                class_name = path.stem.split("_")[0]
                if class_name not in self.classes:
                    self.classes[class_name] = len(self.classes)
        
        print(f"Found {len(self.gspaths)} GS images belonging to {len(self.classes)} classes")
        self.return_paths = return_paths
        self.transform = transform

    def __len__(self):
        return len(self.gspaths)
    
    def __getitem__(self, idx):
        path = self.gspaths[idx]
        class_id = self.classes[path.stem.split("_")[0]]
        sd = torch.load(path)
        gsimage = GSImage.from_statedict(sd)
        if self.transform:
            gsimage = self.transform(gsimage)
        if self.return_paths:
            return gsimage, torch.Tensor([class_id]).to(torch.long), str(path)
        else:
            return gsimage, torch.Tensor([class_id]).to(torch.long)
        
    @staticmethod
    def collate_fn(args):
        ret = []
        for group in zip(*args):
            if isinstance(group[0], GSImage):
                ret.append(BatchedGSImage.from_gsimages(group))
            elif isinstance(group[0], torch.Tensor):
                ret.append(torch.stack(group, 0))
            else:
                ret.append(group)
        return ret


def normalize_values(gs):
    gs.rot.data = gs.rot.data % torch.pi
    return gs


# TODO: quantization-aware image compression !!!
class GSQuantizer():
    def __init__(self,
                 xyrange=(-0.1, 1.1),
                 xysteps=4000,
                 scalerange=(0.1, 2.),
                 scalesteps=1000,
                 rotrange=(0, 3.141593),
                 rotsteps=1000,
                 featrange=(-1., 2.),
                 featsteps=3*256,
                 inverse_scale=True,
                 ):
        self.xyrange = xyrange
        self.xysteps = xysteps
        self.scalerange = scalerange
        self.scalesteps = scalesteps
        self.rotrange = rotrange
        self.rotsteps = rotsteps
        self.featrange = featrange
        self.featsteps = featsteps
        self.inverse_scale = inverse_scale
        if self.inverse_scale:
            self.scalerange = (1/self.scalerange[1], 1/self.scalerange[0])

        # self.xydict = {v: float(k) for k, v in zip(np.linspace(self.xyrange[0], self.xyrange[1], self.xysteps), range(self.xysteps))}
        # self.scaledict = {v: float(k) for k, v in zip(np.linspace(self.scalerange[0], self.scalerange[1], self.scalesteps), range(self.scalesteps))}
        # self.rotdict = {v: float(k) for k, v in zip(np.linspace(self.rotrange[0], self.rotrange[1], self.rotsteps), range(self.rotsteps))}
        # self.featdict = {v: float(k) for k, v in zip(np.linspace(self.featrange[0], self.featrange[1], self.featsteps), range(self.featsteps))}

    def quantize(self, gs):
        # out = GSImage(gs.xy.shape[0], gs.feat.shape[-1], size=gs.size, dtype=torch.long, device=gs.xy.device)
        xy = ((gs.xy - self.xyrange[0]) / (self.xyrange[1] - self.xyrange[0]) * self.xysteps).to(torch.long)
        cscale = 1/gs.scale if self.inverse_scale else gs.scale
        scale = ((cscale - self.scalerange[0]) / (self.scalerange[1] - self.scalerange[0]) * self.scalesteps).to(torch.long)
        rot = ((gs.rot - self.rotrange[0]) / (self.rotrange[1] - self.rotrange[0]) * self.rotsteps).to(torch.long)
        feat = ((gs.feat - self.featrange[0]) / (self.featrange[1] - self.featrange[0]) * self.featsteps).to(torch.long)
        ret = QuantizedGSImage(xy, scale, rot, feat, size=gs.size)
        return ret
    
    def dequantize(self, qgs):
        xy = (qgs.xy.float() / self.xysteps) * (self.xyrange[1] - self.xyrange[0]) + self.xyrange[0]
        scale = (qgs.scale.float() / self.scalesteps) * (self.scalerange[1] - self.scalerange[0]) + self.scalerange[0]
        scale = 1/scale if self.inverse_scale else scale
        rot = (qgs.rot.float() / self.rotsteps) * (self.rotrange[1] - self.rotrange[0]) + self.rotrange[0]
        feat = (qgs.feat.float() / self.featsteps) * (self.featrange[1] - self.featrange[0]) + self.featrange[0]
        ret = GSImage.from_statedict({"scale": scale, "rot": rot, "feat": feat, "xy": xy, "size": qgs.size})
        return ret



class QuantizedGSImage(torch.nn.Module):
    def __init__(self, xy, scale, rot, feat, size):
        super(QuantizedGSImage, self).__init__()
        self.img_h, self.img_w = size
        self.register_buffer("size", torch.LongTensor(size))
        self.xy, self.scale, self.rot, self.feat = xy, scale, rot, feat



def main(rootdir="miniimagenet_256_gs_v2/num=4000_psnr=32/", batch_size=32, num_workers=0):
    
    args = yaml.safe_load(open(Path(rootdir)/"args.yaml", "r"))

    args = SimpleNamespace(**args["cfg"])
    gs = GaussianSplatting2D(args)
    
    q = GSQuantizer()

    ds = ImageNetGSDataset(rootdir)

    img, cls = ds[1]
    qimg = q.quantize(img)
    dqimg = q.dequantize(qimg)
    to_pil_image(gs.forward(dqimg)[0]).save("test2.png")
    print(f"Image shape: {img.xy.shape}, Class ID: {cls}")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=ImageNetGSDataset.collate_fn)
    batch = next(iter(dl))



    ret = normalize_values(batch[0])

    print(ret)


def collate_fn(args):
    return args


def convert(
        rootdir="/home/lukovdg1/miniimagenet_256", 
        num_workers=4,
        num_streams=4,
        from_idx=0,
        to_idx=-1,
        config="cfgs/default_improved.yaml",
        outdir="miniimagenet_256_gs_v2",
        num_gaussians=4000,
        min_steps=700,
        max_steps=2500,
        target_psnr=32,
        imsize=256,
        batsize=1,
        numworkers=4,
        save=True,
        overwrite=False,

    ):
    localargs = locals().copy()
    outpath = Path(outdir) / f"num={num_gaussians}_psnr={target_psnr}"
    outpath.mkdir(parents=True, exist_ok=True)

    gs_dir, recons_dir, diff_dir, orig_dir = outpath / "gs", outpath / "recons", outpath / "diff", outpath / "orig"
    gs_dir.mkdir(parents=True, exist_ok=True); recons_dir.mkdir(parents=True, exist_ok=True); diff_dir.mkdir(parents=True, exist_ok=True); orig_dir.mkdir(parents=True, exist_ok=True)

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
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
    ])
    ds = ImageNetDataset(rootdir, transform=transform, return_paths=True, from_idx=from_idx, to_idx=to_idx)
    print(f"Number of examples: {len(ds)}")
    dataloader = DataLoader(ds, batch_size=batsize, shuffle=False, num_workers=numworkers)

    # initialize ImageGS
    gs = GaussianSplatting2D(cfg)

    print("initialized")
    plot_images = False
    test_reload = True

    for batch in tqdm.tqdm(dataloader):
        images, classes, paths = batch      # (B, C, H, W), in range (0, 1)
        images = images.to(device)
        # impath = Path(paths[0])
        allexist = all([(gs_dir / f"{Path(p).stem}.pt").exists() for p in paths])
        if not overwrite and allexist:
            # print("image exists")
            continue
        gsimages, recons_images, metrics = gs.optimize(images, total_num_gaussians=num_gaussians, target_psnr=target_psnr)
        # ret = optimize_parallel(gs, images, total_num_gaussians=num_gaussians, target_psnr=target_psnr, num_streams=num_streams)
        if save:
            for gsimage, recons_image, image, impath in zip(gsimages, recons_images, images, paths):
                impath = Path(impath)
                torch.save(gsimage.state_dict(), gs_dir / f"{impath.stem}.pt")
                to_pil_image(recons_image.clamp_(0, 1)).save(recons_dir / f"{impath.name}")
                # to_pil_image((image - recons_image).abs().clamp_(0, 1)).save(diff_dir / f"diff_{impath.name}")
                to_pil_image((image).abs().clamp_(0, 1)).save(orig_dir / f"{impath.name}")

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


def optimize_parallel(gs, images, total_num_gaussians=None, target_psnr=None, num_streams=None):
    streams = [torch.cuda.Stream() for _ in range(min(num_streams, len(images)))]
    outputs = [None] * len(images)

    for i, x in enumerate(images):
        stream = streams[i % num_streams]
        with torch.cuda.stream(stream):
            with torch.no_grad():
                outputs[i] = gs.optimize(x.unsqueeze(0), total_num_gaussians=total_num_gaussians, target_psnr=target_psnr)  # one sample at a time

    # Wait for all to finish
    torch.cuda.synchronize()
    return outputs

    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
        # convert()
    else:
        fire.Fire({
            "main": main,
            "convert": convert,
        })
    fire.Fire(main)