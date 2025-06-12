## File for defining the functions for training the 2.5D model.
import torch.nn.functional as F
import torch 
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from pathlib import Path
import sys



class WindowedVolDataset(Dataset):
    """
    Cada amostra:
        x -> FloatTensor [win, H, W]
        y -> LongTensor  [H, W]
    """
    def __init__(self, images_dir, masks_dir,
                 win=11, preload=False, transforms=None):
        assert win % 2 == 1, "win deve ser ímpar"
        self.half     = win // 2
        self.win      = win
        self.preload  = preload
        self.trf      = transforms

        self.img_paths = sorted(Path(images_dir).glob("*.nii*"))
        self.msk_paths = [Path(masks_dir) / p.name for p in self.img_paths]

        if not self.img_paths:
            raise ValueError("Nenhum arquivo .nii/.nii.gz em images_dir")

        # ---------------- índice global ---------------------------------
        self.index = []                # (case_idx, z)
        self.depths = []               # profundidade de cada volume

        for idx, p in enumerate(self.img_paths):
            depth = self._get_depth(p)
            self.depths.append(depth)
            for z in range(self.half, depth - self.half):
                self.index.append((idx, z))

        if not self.index:
            raise ValueError(
                f"Todos os volumes têm depth <= {self.win}. "
                "Reduza win ou verifique os arquivos."
            )

        # opcional: pré-carrega
        if preload:
            self.buffer = {}
            for idx, (ip, mp) in enumerate(zip(self.img_paths,
                                               self.msk_paths)):
                self.buffer[idx] = {
                    "img": self._load_nii(ip, as_mask=False),
                    "mask": self._load_nii(mp, as_mask=True)
                }

    # -------------------------------------------------------------------
    @staticmethod
    def _load_nii(path: Path, as_mask=False):
        arr = nib.load(path).get_fdata(dtype=np.float32)
        # converte máscara p/ int
        if as_mask:
            arr = arr.astype(np.int16)
        # move depth para axis 0 se necessário
        if arr.shape[0] not in (arr.shape[1], arr.shape[2]):
            # depth já é axis 0 -> (D,H,W)
            return arr
        # senão, depth deve ser axis 2 (H,W,D)
        return np.moveaxis(arr, 2, 0)

    @staticmethod
    def _get_depth(path: Path) -> int:
        shape = nib.load(path).shape
        # depth é o eixo cujo tamanho difere dos outros 2 (H,W)
        if shape[0] != shape[1]:
            return shape[0]          # (D,H,W)
        return shape[2]              # (H,W,D)

    # -------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        case_idx, z = self.index[i]

        vol  = (self.buffer[case_idx]["img"]  if self.preload
            else self._load_nii(self.img_paths[case_idx], False))
        mask = (self.buffer[case_idx]["mask"] if self.preload
            else self._load_nii(self.msk_paths[case_idx], True))

        x = vol[z - self.half : z + self.half + 1]   # (win, H, W)
        y = mask[z]                                  # (H, W)

        # ───────────── normalização ─────────────
        # z-score por volume (ou por janela, como preferir)
        mu  = x.mean()
        std = x.std() + 1e-6        # evita divisão por zero
        x   = (x - mu) / std
        # ----------------------------------------

        if self.trf:                # aug / albumentations
            x, y = self.trf(x, y)

        # tensor e quebra do vínculo NumPy
        x = torch.from_numpy(x).float().clone()   # [win, H, W]
        y = torch.from_numpy(y).long().clone()    # [H, W]
        return x, y

def get_dataloaders(path,win_size):
    train_ds = WindowedVolDataset(
        images_dir=f"{path}/imagesTr",
        masks_dir =f"{path}/labelsTr",
        win=win_size,
        preload=True,
        transforms=None          # insira normalização/augment aqui
    )

    validation_ds = WindowedVolDataset(
        images_dir=f"{path}/ImagesVl",
        masks_dir =f"{path}/labelsVl",
        win=win_size,
        preload=True,
        transforms=None          # insira normalização/augment aqui
    )
    return train_ds, validation_ds

def get_dataloaders_resized (path, win_size, batch_size):
    train_ds, validation_ds = get_dataloaders(path, win_size)
    def resize_collate(batch, target_hw=(64, 64)):
        xs, ys = zip(*batch)
        new_h, new_w = target_hw

        xs_resized = [
            F.interpolate(x.unsqueeze(0), size=(new_h, new_w),
                          mode='bilinear', align_corners=False).squeeze(0)
            for x in xs
        ]

        ys_resized = []
        for y in ys:
            had_channel = (y.ndim == 3)
            if y.ndim == 2:
                y = y.unsqueeze(0)

            # interpolate only supports floating dtypes
            y_f = y.float().unsqueeze(0)
            y_r = F.interpolate(y_f, size=(new_h, new_w), mode='nearest').squeeze(0)

            y_r = y_r.to(y.dtype)
            if not had_channel:
                y_r = y_r.squeeze(0)
            ys_resized.append(y_r)

        return torch.stack(xs_resized), torch.stack(ys_resized)

    # DataLoader usando o novo collate
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=resize_collate          # <- aqui!
    )

    val_loader = torch.utils.data.DataLoader(
        validation_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=resize_collate          # <- aqui!
    )
    return loader, val_loader 

def train_2_5D(path, win_size, file_name, LR = 1e-3, N_EPOCHS = 50, BATCH_SIZE  = 4, THRESH_IoU  = 0.20):
    
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, val_loader = get_dataloaders_resized(path,win_size, BATCH_SIZE)

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None).to(DEVICE)


    focal = smp.losses.FocalLoss(mode="binary", alpha=0.9, gamma=2.5) #Up-weighing alpha because of rarity of event
    dice = smp.losses.DiceLoss(mode="binary")

    def loss_fn(pred, target):
        return 0.6 * dice(pred, target) + 0.4 * focal(pred, target)


    pos_weight = torch.tensor([124.], device=DEVICE)  # ≈ 0.992 / 0.008
    bce        = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def soft_dice_loss(logits, target, eps=1e-6):
        probs = logits.sigmoid()
        num = 2 * (probs * target).sum(dim=(2,3))
        den = (probs + target).sum(dim=(2,3)) + eps
        return 1 - (num / den).mean()

    def focal_bce_loss(logits, target, alpha=0.8, gamma=2.5):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * target + (1 - probs) * (1 - target)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_weight = alpha_t * (1 - pt) ** gamma
        return (focal_weight * bce).mean()

    def loss_fn(logits, target):
        return 0.6 * soft_dice_loss(logits, target) + 0.4 * focal_bce_loss(
            logits, target.float())


    metric = BinaryJaccardIndex().to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR)
    scaler    = torch.amp.GradScaler('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    def move_to_device(batch):
        xb, yb = batch
        return xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)

    def resize_logits(logits, target_hw):
        if logits.shape[-2:] == target_hw:
            return logits
        return F.interpolate(logits, size=target_hw,
                             mode="bilinear", align_corners=False)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        metric.reset()
        for batch in loader:
            xb, yb = move_to_device(batch)
            with torch.cuda.amp.autocast():
                logits = resize_logits(model(xb), yb.shape[-2:])
            preds = logits.sigmoid() > THRESH_IoU
            metric.update(preds, yb.unsqueeze(1))
        return metric.compute().item()

    max_score = 0.36
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        i = 0

        for xb, yb in loader:
            xb, yb = move_to_device((xb, yb))

            with torch.amp.autocast('cuda'):
                logits = resize_logits(model(xb), yb.shape[-2:])
                loss   = loss_fn(logits, yb.unsqueeze(1).float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)
            i += 1
            print("\r" + f"epoch:{epoch}, {i}/{4647}", end="")
            if i >= 4647:
              print("")
              break


        train_loss = running_loss / len(loader.dataset)
        val_iou    = evaluate(val_loader)

        print(f"Epoch {epoch:03d} | Loss {train_loss:.4f} | IoU val {val_iou:.4f}")
        if val_iou > max_score:
          torch.save(model, f'best_model{file_name}.p th')
          max_score = val_iou
    return f'best_model{file_name}.p'

def train2D(path, LR = 1e-3, N_EPOCHS = 50, BATCH_SIZE  = 4, THRESH_IoU  = 0.20):
    return train_2_5D(path, 1, "2D", LR, N_EPOCHS, BATCH_SIZE, THRESH_IoU)

if __name__ == "__main__":
    train2D("../../Dataset001_BREAST")