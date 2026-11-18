import os
import sys
import time
import hashlib
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.utils.data import DataLoader, Dataset, Subset

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Model / normalisation / C&W / LR schedule are IDENTICAL to the validated train_eval_full.py ("full_v3_stable_norm_cw").
CODE_VERSION = "mia_v1_from_full_v3"


class TrainingDiverged(RuntimeError):
    pass


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def atomic_npy_save(arr, path):
    tmp_path = f"{path}.tmp.{os.getpid()}.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, path)


def atomic_npz_save(path, **arrays):
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


def atomic_csv_save(df, path):
    tmp_path = f"{path}.tmp.{os.getpid()}.csv"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def split_hash(arr):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(arr, dtype=np.int64)).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- model (same as train_eval_full.py)
class LayerNorm2d(nn.Module):
    """Per-pixel LayerNorm over channels (ConvNeXt style). Strictly per-sample, no running stats."""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class NormalizedNet(nn.Module):
    """Channel normalization as the first layer: inputs live in [0,1] pixel space (same space as d_adv)."""
    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def get_norm_layer(norm_type):
    if norm_type == 'ln':
        return LayerNorm2d
    if norm_type == 'lnchw':     # LayerNorm over (C,H,W) == GroupNorm with 1 group
        return lambda c: nn.GroupNorm(1, c)
    if norm_type == 'gn':
        return lambda c: nn.GroupNorm(min(32, c), c)
    if norm_type == 'bn':
        return nn.BatchNorm2d
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _norm_weight(mod):
    if isinstance(mod, LayerNorm2d):
        return mod.norm.weight
    return getattr(mod, 'weight', None)


def zero_init_last_norm(net):
    for m in net.modules():
        last = m.bn3 if isinstance(m, Bottleneck) else (m.bn2 if isinstance(m, BasicBlock) else None)
        if last is not None:
            w = _norm_weight(last)
            if w is not None:
                nn.init.zeros_(w)


def build_model(model_name, num_classes, norm_type, mean, std, zero_init_res=True):
    factory = {'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50}
    net = factory[model_name](weights=None, norm_layer=get_norm_layer(norm_type))
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if zero_init_res:
        zero_init_last_norm(net)
    return NormalizedNet(net, mean, std)


# ----------------------------------------------------------------------------- data
class IndexedTransformedDataset(Dataset):
    """Transforms applied on the fly (fresh augmentation every epoch); returns the GLOBAL dataset index."""
    def __init__(self, raw_dataset, transform):
        self.raw_dataset = raw_dataset
        self.transform = transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        img, target = self.raw_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return idx, img, target


def get_dataset_stats(dataset_name):
    if dataset_name == 'cifar10':
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10, datasets.CIFAR10
    if dataset_name == 'cifar100':
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100, datasets.CIFAR100
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def setup_and_save_splits(out_dir, dataset, model, total_samples, eval_size, num_shadows, split_seed=42):
    """Consistent 25k member / 25k non-member split, shadow splits and C&W eval subset (shared across models)."""
    half_size = total_samples // 2
    canonical_file = os.path.join(out_dir, f"splits_{dataset}.pt")
    model_split_file = os.path.join(out_dir, f"splits_{dataset}_{model}.pt")

    splits = None
    if os.path.exists(canonical_file):
        splits = torch.load(canonical_file, map_location='cpu', weights_only=False)
    elif os.path.exists(model_split_file):
        splits = torch.load(model_split_file, map_location='cpu', weights_only=False)

    updated = False
    if splits is None:
        rng_main = np.random.RandomState(split_seed)
        in_member_split = np.sort(rng_main.choice(total_samples, half_size, replace=False)).astype(np.int64)
        non_member_split = np.setdiff1d(np.arange(total_samples, dtype=np.int64), in_member_split)
        shadow_splits = []
        for k in range(max(num_shadows, 64)):
            rng_s = np.random.RandomState(split_seed + 10000 + k)
            shadow_splits.append(np.sort(rng_s.choice(total_samples, half_size, replace=False)).astype(np.int64))
        splits = {'dataset': dataset, 'split_seed': split_seed, 'total_samples': total_samples,
                  'half_size': half_size, 'in_member_split': in_member_split,
                  'non_member_split': non_member_split, 'shadow_splits': shadow_splits, 'eval_Subsets': {}}
        updated = True
    else:
        in_member_split = np.asarray(splits['in_member_split'], dtype=np.int64)
        assert len(in_member_split) == half_size, f"Expected {half_size} in-members, got {len(in_member_split)}"
        if 'non_member_split' not in splits:
            splits['non_member_split'] = np.setdiff1d(np.arange(total_samples, dtype=np.int64), in_member_split)
            updated = True
        if 'eval_Subsets' not in splits:
            splits['eval_Subsets'] = {}
            updated = True
        existing = len(splits['shadow_splits'])
        if existing < num_shadows:
            for k in range(existing, num_shadows):
                rng_s = np.random.RandomState(split_seed + 10000 + k)
                splits['shadow_splits'].append(np.sort(rng_s.choice(total_samples, half_size, replace=False)).astype(np.int64))
            updated = True

    eff_eval_size = half_size if (eval_size <= 0 or eval_size >= half_size) else int(eval_size)
    if eff_eval_size not in splits['eval_Subsets']:
        if eff_eval_size == half_size:
            eval_local_idx = np.arange(half_size, dtype=np.int64)
        else:
            rng_eval = np.random.RandomState(split_seed + 500)
            eval_local_idx = np.sort(rng_eval.choice(half_size, size=eff_eval_size, replace=False)).astype(np.int64)
        splits['eval_Subsets'][eff_eval_size] = {'eval_local_idx': eval_local_idx,
                                                 'eval_global_idx': in_member_split[eval_local_idx]}
        updated = True

    if updated or not os.path.exists(canonical_file):
        atomic_torch_save(splits, canonical_file)
    if updated or not os.path.exists(model_split_file):
        atomic_torch_save(splits, model_split_file)
    in_member_npy = os.path.join(out_dir, f"in_member_25k_indices_{dataset}.npy")
    if not os.path.exists(in_member_npy):
        atomic_npy_save(in_member_split, in_member_npy)

    ev = splits['eval_Subsets'][eff_eval_size]
    return (in_member_split, np.asarray(splits['non_member_split'], dtype=np.int64),
            [np.asarray(s, dtype=np.int64) for s in splits['shadow_splits'][:num_shadows]],
            np.asarray(ev['eval_local_idx'], dtype=np.int64), np.asarray(ev['eval_global_idx'], dtype=np.int64))


# ----------------------------------------------------------------------------- C&W (same as train_eval_full.py)
def compute_cw_adv_distance(model, x, y, max_eps=6.0, cw_bs_steps=5, cw_iters=200, cw_lr=0.01, init_const=0.01):
    """Untargeted L2 C&W in tanh space; x in [0,1]. Failed/censored samples: d=max_eps, success=False."""
    model.eval()
    prev_grad_state = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        B, device = x.size(0), x.device
        d_adv = torch.full((B,), float(max_eps), dtype=torch.float32, device=device)
        success = torch.zeros(B, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits0 = model(x)
            if not torch.isfinite(logits0).all():
                raise FloatingPointError("Model produces non-finite logits; the checkpoint is broken (diverged training).")
            init_mis = logits0.argmax(dim=1) != y
        d_adv[init_mis] = 0.0
        success[init_mis] = True

        idx = torch.where(~init_mis)[0]
        if idx.numel() == 0:
            return d_adv.cpu().numpy(), success.cpu().numpy(), init_mis.cpu().numpy()

        x0 = x[idx].clamp(0.0, 1.0)
        y0 = y[idx]
        n, K = x0.size(0), logits0.size(1)
        w0 = torch.atanh((2.0 * x0 - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        ar = torch.arange(n, device=device)
        true_mask = torch.zeros(n, K, dtype=torch.bool, device=device)
        true_mask[ar, y0] = True

        const = torch.full((n,), float(init_const), device=device)
        lo = torch.zeros(n, device=device)
        hi = torch.full((n,), float('inf'), device=device)
        best_l2 = torch.full((n,), float('inf'), device=device)

        for _ in range(cw_bs_steps):
            w = w0.clone().requires_grad_(True)
            opt = optim.Adam([w], lr=cw_lr)
            succ_c = torch.zeros(n, dtype=torch.bool, device=device)
            for _ in range(cw_iters):
                x_adv = 0.5 * (torch.tanh(w) + 1.0)
                logits = model(x_adv)
                sq_l2 = (x_adv - x0).pow(2).flatten(1).sum(dim=1)
                real = logits[ar, y0]
                other = logits.masked_fill(true_mask, float('-inf')).amax(dim=1)
                f = torch.clamp(real - other, min=0.0)
                loss = (sq_l2 + const * f).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite logits during C&W.")
                    l2 = sq_l2.sqrt()
                    is_adv = logits.argmax(dim=1) != y0
                    improved = is_adv & (l2 < best_l2)
                    best_l2 = torch.where(improved, l2, best_l2)
                    succ_c |= is_adv
                opt.step()
            with torch.no_grad():
                hi = torch.where(succ_c, torch.minimum(hi, const), hi)
                lo = torch.where(~succ_c, torch.maximum(lo, const), lo)
                const = torch.where(torch.isfinite(hi), 0.5 * (lo + hi), const * 10.0)

        valid = torch.isfinite(best_l2) & (best_l2 <= max_eps)
        d_adv[idx] = torch.where(valid, best_l2, torch.full_like(best_l2, float(max_eps)))
        success[idx] = valid
        return d_adv.cpu().numpy(), success.cpu().numpy(), init_mis.cpu().numpy()
    finally:
        for p, state in zip(model.parameters(), prev_grad_state):
            p.requires_grad_(state)


# ----------------------------------------------------------------------------- eval / stats
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _, x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                return float('nan')
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def derive_stats(epoch_losses, epoch_grads, L_u, epochs):
    el = epoch_losses.astype(np.float64)
    eg = epoch_grads.astype(np.float64)
    csl, csg = el.sum(axis=0), eg.sum(axis=0)
    final_loss, final_grad = el[-1], eg[-1]
    w40, w50 = max(1, int(0.40 * epochs)), max(1, int(0.50 * epochs))
    q75, q25 = np.percentile(el, [75, 25], axis=0)
    return {
        'csl': csl, 'csg': csg, 'final_loss': final_loss, 'final_grad': final_grad,
        'csl_diff_40': el[:w40].sum(axis=0) - el[-w40:].sum(axis=0),
        'csl_diff_50': el[:w50].sum(axis=0) - el[-w50:].sum(axis=0),
        'lt_mean': el.mean(axis=0), 'lt_std': el.std(axis=0), 'lt_iqr': q75 - q25,
        'd_adv_proxy': 2.0 * np.maximum(0.0, L_u * epochs - csl) / np.maximum(csg, 1e-12),
        'd_adv_1_proxy': 2.0 * np.maximum(0.0, L_u - final_loss) / np.maximum(final_grad, 1e-12),
    }


def compute_mia_scores(shadow_losses, shadow_in_mask):
    """Vectorised per-sample Yeom and LiRA advantages (same formulas as the original per-sample loop).
    shadow_losses: (K, n) losses of K shadows on the n target members; shadow_in_mask: (K, n) bool."""
    L = np.clip(shadow_losses.astype(np.float64), 0.0, 1e4)
    M = shadow_in_mask.astype(bool)
    Mo = ~M
    n_in = M.sum(axis=0).astype(np.float64)
    n_out = Mo.sum(axis=0).astype(np.float64)
    valid = (n_in > 0) & (n_out > 0)
    ni, no = np.maximum(n_in, 1.0), np.maximum(n_out, 1.0)

    yeom = (np.exp(-L) * M).sum(axis=0) / ni - (np.exp(-L) * Mo).sum(axis=0) / no

    logL = np.log(L + 1e-10)
    mu_in = (logL * M).sum(axis=0) / ni
    mu_out = (logL * Mo).sum(axis=0) / no
    sd_in = np.sqrt((((logL - mu_in) ** 2) * M).sum(axis=0) / ni) + 1e-10
    sd_out = np.sqrt((((logL - mu_out) ** 2) * Mo).sum(axis=0) / no) + 1e-10

    def lognorm(x, mu, sd):   # Gaussian log-pdf without the shared -0.5*log(2*pi) constant (cancels in the ratio)
        return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)

    ratio = lognorm(logL, mu_in, sd_in) - lognorm(logL, mu_out, sd_out)
    sig = expit(ratio)
    lira = (sig * M).sum(axis=0) / ni - (sig * Mo).sum(axis=0) / no
    lira_thr = ((ratio > 0) & M).sum(axis=0) / ni - ((ratio > 0) & Mo).sum(axis=0) / no

    for a in (yeom, lira, lira_thr):
        a[~valid] = np.nan
    return yeom, lira, lira_thr, n_in.astype(np.int64)


def print_cw_attack_statistics(df_eval, dataset, model, seed_label="All Seeds"):
    sub = df_eval.dropna(subset=['d_adv'])
    total = len(sub)
    if total == 0:
        print(f"[!] No C&W results for {dataset}/{model} ({seed_label}).")
        return
    succ = sub['cw_success'].astype(bool).values
    init_mis = sub['init_misclassified'].astype(bool).values
    attacked = ~init_mis
    flipped = attacked & succ
    failed = ~succ
    valid_d = sub['d_adv'].values[succ]
    flipped_d = sub['d_adv'].values[flipped]
    print("------------------------------------------------------------------------------------------")
    print(f" [MIA Target] C&W Attack Statistics | {dataset.upper()} | {model.upper()} | {seed_label}")
    print(f"   Total Evaluated In-Members    : {total}")
    print(f"   Initially Misclassified (d=0) : {init_mis.sum()} ({100.0 * init_mis.mean():.2f}%)")
    print(f"   Correctly Classified Attacked : {attacked.sum()}")
    print(f"   Successfully Flipped by C&W   : {flipped.sum()} (Conditional ASR: {100.0 * flipped.sum() / max(attacked.sum(), 1):.2f}%)")
    print(f"   Failed / Censored (max_eps)   : {failed.sum()} ({100.0 * failed.mean():.2f}%)")
    print(f"   Overall Valid Boundary Rate   : {succ.sum()}/{total} ({100.0 * succ.mean():.2f}%)")
    if len(valid_d) > 0:
        print(f"   Valid d_adv (Mean +/- Std)    : {np.mean(valid_d):.4f} +/- {np.std(valid_d):.4f} (Median: {np.median(valid_d):.4f})")
    if len(flipped_d) > 0:
        print(f"   Flipped-only d_adv (Mean/Med) : {np.mean(flipped_d):.4f} / {np.median(flipped_d):.4f}")
    print("------------------------------------------------------------------------------------------")


# ----------------------------------------------------------------------------- training (same recipe as train_eval_full.py)
def lr_at(it, iters_per_epoch, base_lr, args):
    warm_iters = args.warmup_epochs * iters_per_epoch
    if warm_iters > 0 and it < warm_iters:
        return base_lr * (it + 1) / warm_iters
    ep = it // iters_per_epoch
    factor = 1.0
    if ep >= int(args.epochs * 0.5):
        factor *= 0.1
    if ep >= int(args.epochs * 0.75):
        factor *= 0.1
    return base_lr * factor


def train_model(args, tag, seed, base_lr, train_idx, train_ds, test_loader, device, num_classes,
                total_samples, mean, std, std_t, track, amp, log_every):
    """Trains on the global indices `train_idx`. If track=True, records per-sample loss and exact per-sample
    input-gradient norm every epoch (LN/GN are per-sample => B * d(mean loss)/dx_i = d loss_i / dx_i)."""
    set_seed(seed)
    model = build_model(args.model, num_classes, args.norm_layer, mean, std,
                        zero_init_res=not args.no_zero_init_residual).to(device)
    n = len(train_idx)
    g2l = torch.full((total_samples,), -1, dtype=torch.long, device=device)
    g2l[torch.as_tensor(train_idx, dtype=torch.long, device=device)] = torch.arange(n, device=device)

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, generator=gen,
                        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    track_loss = torch.full((args.epochs, n), float('nan'), device=device) if track else None
    track_grad = torch.full((args.epochs, n), float('nan'), device=device) if track else None
    iters_per_epoch = len(loader)
    chance = 100.0 / num_classes
    it = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        sum_loss = torch.zeros((), device=device)
        n_correct = torch.zeros((), device=device)
        n_seen = 0
        lr = base_lr
        for idxs, x, y in loader:
            lr = lr_at(it, iters_per_epoch, base_lr, args)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if track:
                x.requires_grad_(True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(x)
            losses = criterion(out.float(), y)
            mean_loss = losses.mean()
            if not torch.isfinite(mean_loss):
                raise TrainingDiverged(f"{tag}: non-finite loss at epoch {epoch + 1}, iter {it}, lr={lr:.4g}")
            optimizer.zero_grad(set_to_none=True)
            mean_loss.backward()

            with torch.no_grad():
                B = x.size(0)
                if track:
                    g_in = x.grad * B                       # [0,1] pixel space (same space as d_adv)
                    if args.grad_space == 'norm':
                        g_in = g_in * std_t
                    loc = g2l[idxs.to(device, non_blocking=True)]
                    track_loss[epoch, loc] = losses.detach()
                    track_grad[epoch, loc] = g_in.flatten(1).norm(dim=1)
                sum_loss += losses.detach().sum()
                n_correct += (out.argmax(dim=1) == y).sum()
                n_seen += B

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            it += 1

        if track and (torch.isnan(track_loss[epoch]).any() or torch.isnan(track_grad[epoch]).any()):
            raise RuntimeError(f"{tag}: some samples received no loss/gradient record in epoch {epoch + 1}.")
        train_acc = 100.0 * n_correct.item() / n_seen
        last = (epoch + 1 == args.epochs)
        if epoch == 0 or (epoch + 1) % log_every == 0 or last:
            msg = (f"[{tag}] Epoch {epoch + 1:3d}/{args.epochs} | lr {lr:.5f} | train loss {sum_loss.item() / n_seen:.4f} "
                   f"| train acc {train_acc:.2f}% | {time.time() - t0:.1f}s/epoch")
            if (epoch + 1) % args.eval_every == 0 or last:
                msg += f" | test acc {evaluate_accuracy(model, test_loader, device):.2f}%"
            print(msg, flush=True)
        if args.sanity_epoch > 0 and epoch + 1 == args.sanity_epoch and train_acc < 2.0 * chance:
            raise TrainingDiverged(f"{tag}: train acc {train_acc:.2f}% after {epoch + 1} epochs is ~chance ({chance:.1f}%)")

    if track:
        return model, track_loss.cpu().numpy(), track_grad.cpu().numpy()
    return model, None, None


def train_with_retries(args, tag, seed, train_idx, train_ds, test_loader, device, num_classes,
                       total_samples, mean, std, std_t, track, amp, log_every, min_test_acc):
    for attempt in range(args.max_retries + 1):
        lr = args.lr / (2 ** attempt)
        try:
            model, tl, tg = train_model(args, tag, seed, lr, train_idx, train_ds, test_loader, device, num_classes,
                                        total_samples, mean, std, std_t, track, amp, log_every)
            acc = evaluate_accuracy(model, test_loader, device)
            if not np.isfinite(acc) or acc < min_test_acc:
                raise TrainingDiverged(f"{tag}: final test accuracy {acc} < {min_test_acc}")
            return model, tl, tg, acc, lr
        except TrainingDiverged as e:
            print(f"[!] {e}", flush=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if attempt == args.max_retries:
                raise RuntimeError(f"{tag}: training failed after {args.max_retries + 1} attempts.")
            print(f"[*] Retrying {tag} with lr={args.lr / (2 ** (attempt + 1))}", flush=True)


# ----------------------------------------------------------------------------- target phase
def load_valid_target_ckpt(path, args, in_member_split, min_test_acc):
    if args.force_retrain_targets or not os.path.exists(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    reasons = []
    for key, want in (('code_version', CODE_VERSION), ('epoch', args.epochs), ('norm_layer', args.norm_layer),
                      ('grad_space', args.grad_space)):
        if ck.get(key) != want:
            reasons.append(f"{key}={ck.get(key)}")
    if not np.array_equal(np.asarray(ck.get('in_member_split', [])), in_member_split):
        reasons.append("split mismatch")
    acc = ck.get('test_acc', float('nan'))
    if not np.isfinite(acc) or acc < min_test_acc:
        reasons.append(f"test_acc={acc}")
    if 'state_dict' in ck and not all(torch.isfinite(v).all() for v in ck['state_dict'].values() if v.is_floating_point()):
        reasons.append("non-finite weights")
    if reasons:
        print(f"[!] Ignoring existing target checkpoint {path} ({', '.join(reasons)}); retraining.")
        return None
    return ck


def run_target_seed(seed, args, ctx):
    in_member_split = ctx['in_member_split']
    half = len(in_member_split)
    device = ctx['device']
    path = os.path.join(ctx['ckpt_target_dir'], f"target_seed_{seed}_final.pth")
    g2l_np = np.full(ctx['total_samples'], -1, dtype=np.int64)
    g2l_np[in_member_split] = np.arange(half)

    ck = load_valid_target_ckpt(path, args, in_member_split, ctx['min_test_acc'])
    if ck is not None:
        print(f"[*] Loaded verified target checkpoint for seed {seed} (test acc {ck['test_acc']:.2f}%). Skipping training.")
        model = build_model(args.model, ctx['num_classes'], args.norm_layer, ctx['mean'], ctx['std'],
                            zero_init_res=not args.no_zero_init_residual).to(device)
        model.load_state_dict(ck['state_dict'])
        epoch_losses, epoch_grads = ck['epoch_losses'], ck['epoch_grads']
        test_acc, holdout_acc, lr_used = ck['test_acc'], ck.get('holdout_acc', float('nan')), ck.get('lr_used', args.lr)
        d_adv, cw_succ, init_mis = ck['d_adv_25k'].copy(), ck['cw_success_25k'].copy(), ck['init_mis_25k'].copy()
    else:
        print(f"\n[*] [MIA 25k Target] {args.model} ({args.norm_layer}) | {args.dataset} | Seed {seed} | "
              f"C&W eval size {len(ctx['eval_local_idx'])}")
        model, epoch_losses, epoch_grads, test_acc, lr_used = train_with_retries(
            args, f"Target s{seed}", seed, in_member_split, ctx['train_ds'], ctx['test_loader'], device,
            ctx['num_classes'], ctx['total_samples'], ctx['mean'], ctx['std'], ctx['std_t'],
            track=True, amp=False, log_every=1, min_test_acc=ctx['min_test_acc'])
        holdout_acc = evaluate_accuracy(model, ctx['holdout_loader'], device)
        print(f"[*] Target seed {seed} done | lr {lr_used} | test acc {test_acc:.2f}% | non-member holdout acc {holdout_acc:.2f}%")
        d_adv = np.full(half, np.nan, dtype=np.float64)
        cw_succ = np.zeros(half, dtype=bool)
        init_mis = np.zeros(half, dtype=bool)

    def save_ckpt():
        atomic_torch_save({
            'code_version': CODE_VERSION, 'epoch': args.epochs, 'seed': seed, 'model': args.model,
            'dataset': args.dataset, 'norm_layer': args.norm_layer, 'grad_space': args.grad_space,
            'lr_used': lr_used, 'test_acc': test_acc, 'holdout_acc': holdout_acc,
            'in_member_split': in_member_split, 'state_dict': model.state_dict(),
            'epoch_losses': epoch_losses, 'epoch_grads': epoch_grads,
            'd_adv_25k': d_adv, 'cw_success_25k': cw_succ, 'init_mis_25k': init_mis,
        }, path)

    if ck is None:
        save_ckpt()   # persist training before the long C&W stage

    eval_local_idx = ctx['eval_local_idx']
    todo_local = eval_local_idx[np.isnan(d_adv[eval_local_idx])]
    if len(todo_local) > 0:
        todo_global = in_member_split[todo_local]
        print(f"[*] Running C&W on {len(todo_global)} in-members (seed {seed}, batch {args.cw_batch_size})...", flush=True)
        loader = DataLoader(Subset(ctx['eval_ds'], todo_global.tolist()), batch_size=args.cw_batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for b, (idxs, x, y) in enumerate(loader):
            t0 = time.time()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bd, bs, bm = compute_cw_adv_distance(model, x, y, max_eps=args.max_eps, cw_bs_steps=args.cw_bs_steps,
                                                 cw_iters=args.cw_iters, cw_lr=args.cw_lr, init_const=args.cw_init_const)
            loc = g2l_np[idxs.numpy()]
            assert (loc >= 0).all()
            d_adv[loc], cw_succ[loc], init_mis[loc] = bd, bs, bm
            print(f"    C&W batch {b + 1}/{len(loader)} | {time.time() - t0:.1f}s | success {100.0 * bs.mean():.1f}%", flush=True)
            if (b + 1) % args.cw_save_every == 0:
                save_ckpt()
        save_ckpt()

    df = pd.DataFrame({'d_adv': d_adv[eval_local_idx], 'cw_success': cw_succ[eval_local_idx],
                       'init_misclassified': init_mis[eval_local_idx]})
    print_cw_attack_statistics(df, args.dataset, args.model, seed_label=f"Seed {seed}")
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- shadow phase
def shadow_paths(ctx, k):
    return (os.path.join(ctx['ckpt_shadow_dir'], f"shadow_{k + 1}_final.pth"),
            os.path.join(ctx['ckpt_shadow_dir'], f"shadow_{k + 1}_losses.npz"))


def load_valid_shadow_losses(npz_path, args, ctx, k):
    if not os.path.exists(npz_path):
        return None
    try:
        d = np.load(npz_path, allow_pickle=False)
        ok = (str(d['code_version']) == CODE_VERSION and str(d['norm_layer']) == args.norm_layer
              and int(d['epochs']) == args.epochs and str(d['split_hash']) == split_hash(ctx['shadow_splits'][k])
              and str(d['target_hash']) == split_hash(ctx['in_member_split'])
              and d['losses'].shape == (len(ctx['in_member_split']),) and np.isfinite(d['losses']).all())
        return d['losses'].astype(np.float32) if ok else None
    except Exception:
        return None


def run_shadows(s_start, s_end, args, ctx):
    device = ctx['device']
    in_member_split = ctx['in_member_split']
    half = len(in_member_split)
    g2l_np = np.full(ctx['total_samples'], -1, dtype=np.int64)
    g2l_np[in_member_split] = np.arange(half)
    member_loader = DataLoader(Subset(ctx['eval_ds'], in_member_split.tolist()), batch_size=512, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for k in range(s_start, s_end):
        pth, npz = shadow_paths(ctx, k)
        if not args.force_retrain_shadows and load_valid_shadow_losses(npz, args, ctx, k) is not None:
            print(f"[*] Shadow {k + 1}/{args.num_shadows} already trained & evaluated (valid). Skipping.")
            continue

        model, test_acc = None, float('nan')
        if os.path.exists(pth) and not args.force_retrain_shadows:
            st = torch.load(pth, map_location='cpu', weights_only=False)
            if (isinstance(st, dict) and st.get('code_version') == CODE_VERSION and st.get('norm_layer') == args.norm_layer
                    and st.get('epoch') == args.epochs and st.get('split_hash') == split_hash(ctx['shadow_splits'][k])
                    and np.isfinite(st.get('test_acc', float('nan'))) and st['test_acc'] >= ctx['min_test_acc']):
                model = build_model(args.model, ctx['num_classes'], args.norm_layer, ctx['mean'], ctx['std'],
                                    zero_init_res=not args.no_zero_init_residual).to(device)
                model.load_state_dict(st['state_dict'])
                test_acc = st['test_acc']
                print(f"[*] Loaded verified Shadow {k + 1} weights (test acc {test_acc:.2f}%).")
            else:
                print(f"[!] Ignoring stale/invalid shadow weights {pth}; retraining.")

        if model is None:
            t0 = time.time()
            print(f"[*] Training Shadow {k + 1}/{args.num_shadows} ({args.dataset} | {args.model} | "
                  f"{'bf16 AMP' if args.shadow_amp else 'fp32/TF32'})...", flush=True)
            model, _, _, test_acc, lr_used = train_with_retries(
                args, f"Shadow {k + 1}", args.shadow_base_seed + k, ctx['shadow_splits'][k], ctx['train_ds'],
                ctx['test_loader'], device, ctx['num_classes'], ctx['total_samples'], ctx['mean'], ctx['std'],
                ctx['std_t'], track=False, amp=args.shadow_amp, log_every=20, min_test_acc=ctx['min_test_acc'])
            print(f"[*] Shadow {k + 1} done in {(time.time() - t0) / 60:.1f} min | lr {lr_used} | test acc {test_acc:.2f}%")
            if not args.no_save_shadow_weights:
                atomic_torch_save({'code_version': CODE_VERSION, 'shadow_idx': k + 1, 'shadow_seed': args.shadow_base_seed + k,
                                   'norm_layer': args.norm_layer, 'epoch': args.epochs, 'test_acc': test_acc,
                                   'split_hash': split_hash(ctx['shadow_splits'][k]),
                                   'shadow_split': ctx['shadow_splits'][k], 'state_dict': model.state_dict()}, pth)

        model.eval()
        losses_k = np.full(half, np.nan, dtype=np.float32)
        with torch.no_grad():
            for idxs, x, y in member_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                losses_k[g2l_np[idxs.numpy()]] = criterion(model(x), y).float().cpu().numpy()
        if not np.isfinite(losses_k).all():
            raise RuntimeError(f"Shadow {k + 1}: non-finite member losses.")
        atomic_npz_save(npz, losses=losses_k, code_version=np.array(CODE_VERSION), norm_layer=np.array(args.norm_layer),
                        epochs=np.array(args.epochs), test_acc=np.array(test_acc),
                        split_hash=np.array(split_hash(ctx['shadow_splits'][k])),
                        target_hash=np.array(split_hash(in_member_split)))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- aggregate phase
def aggregate_and_save_mia(args, ctx):
    in_member_split = ctx['in_member_split']
    half = len(in_member_split)
    total = ctx['total_samples']
    targets_all = ctx['targets_all']
    L_u = ctx['L_u']

    shadow_losses = np.zeros((args.num_shadows, half), dtype=np.float32)
    shadow_in_mask = np.zeros((args.num_shadows, half), dtype=bool)
    missing = []
    for k in range(args.num_shadows):
        _, npz = shadow_paths(ctx, k)
        l = load_valid_shadow_losses(npz, args, ctx, k)
        if l is None:
            missing.append(k + 1)
            continue
        shadow_losses[k] = l
        full_mask = np.zeros(total, dtype=bool)
        full_mask[ctx['shadow_splits'][k]] = True
        shadow_in_mask[k] = full_mask[in_member_split]
    if missing:
        raise FileNotFoundError(f"Missing/invalid shadow loss files for shadows {missing} in {ctx['ckpt_shadow_dir']}.")

    print(f"\n[*] Computing LiRA and Yeom advantages over {args.num_shadows} shadows for {half} in-members...")
    yeom_adv, lira_adv, lira_thr_adv, n_in = compute_mia_scores(shadow_losses, shadow_in_mask)
    print(f"    shadows-IN per sample: min {n_in.min()}, mean {n_in.mean():.1f}, max {n_in.max()} | "
          f"NaN scores: {int(np.isnan(lira_adv).sum())}")

    eval_local_idx = ctx['eval_local_idx']
    eval_dfs, all_dfs, missing_seeds = [], [], []
    for seed in args.target_seeds_for_merge:
        path = os.path.join(ctx['ckpt_target_dir'], f"target_seed_{seed}_final.pth")
        ck = load_valid_target_ckpt(path, argparse.Namespace(**{**vars(args), 'force_retrain_targets': False}),
                                    in_member_split, ctx['min_test_acc'])
        if ck is None:
            missing_seeds.append(seed)
            continue
        st = derive_stats(ck['epoch_losses'], ck['epoch_grads'], L_u, args.epochs)
        attempted = ~np.isnan(ck['d_adv_25k'])
        df = pd.DataFrame({
            'model': args.model, 'dataset': args.dataset, 'norm_layer': args.norm_layer, 'seed': seed,
            'test_acc': ck['test_acc'], 'lr_used': ck.get('lr_used', args.lr),
            'global_idx': in_member_split, 'in_member_local_idx': np.arange(half, dtype=np.int64),
            'target': targets_all[in_member_split],
            **{c: st[c] for c in ['csl', 'csg', 'final_loss', 'final_grad', 'csl_diff_40', 'csl_diff_50',
                                  'lt_mean', 'lt_std', 'lt_iqr']},
            'd_adv': ck['d_adv_25k'], 'd_adv_proxy': st['d_adv_proxy'], 'd_adv_1_proxy': st['d_adv_1_proxy'],
            'init_misclassified': ck['init_mis_25k'], 'cw_success': ck['cw_success_25k'], 'cw_attempted': attempted,
            'yeom_adv': yeom_adv, 'lira_adv': lira_adv, 'lira_thr_adv': lira_thr_adv, 'n_shadows_in': n_in,
        })
        df_eval = df.iloc[eval_local_idx].copy().reset_index(drop=True)
        print_cw_attack_statistics(df_eval, args.dataset, args.model, seed_label=f"Seed {seed}")
        eval_dfs.append(df_eval)
        all_dfs.append(df)
    if missing_seeds:
        raise FileNotFoundError(f"Missing/invalid target checkpoints for seeds {missing_seeds} in {ctx['ckpt_target_dir']}.")

    comb_eval = pd.concat(eval_dfs, ignore_index=True)
    out_eval_csv = os.path.join(args.out_dir, f"mia_{args.dataset}_{args.model}_3seeds.csv")
    atomic_csv_save(comb_eval, out_eval_csv)
    print(f"[+] Saved MIA C&W evaluation metrics ({len(comb_eval)} rows) to: {out_eval_csv}")
    print_cw_attack_statistics(comb_eval, args.dataset, args.model, seed_label=f"Merged {len(eval_dfs)} Seeds")
    out_all_csv = os.path.join(args.out_dir, f"mia_all25k_dynamics_{args.dataset}_{args.model}_3seeds.csv")
    atomic_csv_save(pd.concat(all_dfs, ignore_index=True), out_all_csv)
    print(f"[+] Saved all 25,000 in-member MIA & trajectory dynamics to: {out_all_csv}")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="MIA half-dataset (25k) target & shadow pipeline with C&W eval (stable LN recipe)")
    p.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
    p.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet50'])
    p.add_argument('--norm-layer', type=str, default='lnchw', choices=['ln', 'lnchw', 'gn', 'bn'])
    p.add_argument('--grad-space', type=str, default='pixel', choices=['pixel', 'norm'])
    p.add_argument('--phase', type=str, default='all', choices=['all', 'init-splits', 'target', 'shadow', 'aggregate'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--target-seeds-for-merge', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--num-shadows', type=int, default=64)
    p.add_argument('--shadow-start', type=int, default=0)
    p.add_argument('--shadow-end', type=int, default=-1)
    p.add_argument('--shadow-base-seed', type=int, default=1000)
    p.add_argument('--shadow-amp', action='store_true', help='bf16 autocast for SHADOW training only (~1.5-2x faster)')
    p.add_argument('--no-save-shadow-weights', action='store_true', help='Keep only shadow losses (saves disk)')
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--eval-size', type=int, default=25000)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--no-zero-init-residual', action='store_true')
    p.add_argument('--max-retries', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--sanity-epoch', type=int, default=10)
    p.add_argument('--min-test-acc', type=float, default=None, help='default 40 (C10) / 20 (C100)')
    p.add_argument('--cw-batch-size', type=int, default=512)
    p.add_argument('--cw-bs-steps', type=int, default=5)
    p.add_argument('--cw-iters', type=int, default=200)
    p.add_argument('--cw-lr', type=float, default=0.01)
    p.add_argument('--cw-init-const', type=float, default=0.01)
    p.add_argument('--cw-save-every', type=int, default=10)
    p.add_argument('--max-eps', type=float, default=6.0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--out-dir', type=str, default='./results')
    p.add_argument('--force-retrain-targets', action='store_true')
    p.add_argument('--force-retrain-shadows', action='store_true')
    args = p.parse_args()

    if args.norm_layer == 'bn':
        print("[!] WARNING: BatchNorm couples samples in a batch; B*x.grad is NOT the per-sample input gradient.")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.model}_{args.norm_layer}"
    ckpt_target_dir = os.path.join(args.out_dir, "checkpoints_target_mia", tag)
    ckpt_shadow_dir = os.path.join(args.out_dir, "checkpoints_shadows", tag)
    os.makedirs(ckpt_target_dir, exist_ok=True)
    os.makedirs(ckpt_shadow_dir, exist_ok=True)

    mean, std, num_classes, ds_cls = get_dataset_stats(args.dataset)
    raw_train = ds_cls(root=args.data_dir, train=True, download=True)
    raw_test = ds_cls(root=args.data_dir, train=False, download=True)
    total_samples = len(raw_train)

    in_member_split, non_member_split, shadow_splits, eval_local_idx, eval_global_idx = setup_and_save_splits(
        args.out_dir, args.dataset, args.model, total_samples, args.eval_size, args.num_shadows, split_seed=args.split_seed)
    if args.phase == 'init-splits':
        print(f"[+] Initialized 25k member split, {args.num_shadows} shadow splits, {len(eval_local_idx)} C&W eval "
              f"indices for {args.dataset} ({args.model}).")
        return

    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    transform_eval = transforms.ToTensor()    # normalization lives inside the model
    eval_ds = IndexedTransformedDataset(raw_train, transform_eval)
    ctx = {
        'device': device, 'num_classes': num_classes, 'mean': mean, 'std': std,
        'std_t': torch.tensor(std, device=device).view(1, 3, 1, 1),
        'L_u': float(np.log(num_classes) + 1e-3),
        'min_test_acc': args.min_test_acc if args.min_test_acc is not None else (40.0 if num_classes == 10 else 20.0),
        'total_samples': total_samples, 'targets_all': np.array(raw_train.targets, dtype=np.int64),
        'in_member_split': in_member_split, 'shadow_splits': shadow_splits,
        'eval_local_idx': eval_local_idx, 'ckpt_target_dir': ckpt_target_dir, 'ckpt_shadow_dir': ckpt_shadow_dir,
        'train_ds': IndexedTransformedDataset(raw_train, transform_train), 'eval_ds': eval_ds,
        'test_loader': DataLoader(IndexedTransformedDataset(raw_test, transform_eval), batch_size=512, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True),
        'holdout_loader': DataLoader(Subset(eval_ds, non_member_split.tolist()), batch_size=512, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True),
    }

    if args.phase in ('all', 'target'):
        for seed in args.seeds:
            run_target_seed(seed, args, ctx)
    if args.phase in ('all', 'shadow'):
        s_start = max(0, args.shadow_start)
        s_end = args.num_shadows if args.shadow_end < 0 else min(args.num_shadows, args.shadow_end)
        run_shadows(s_start, s_end, args, ctx)
    if args.phase in ('all', 'aggregate'):
        aggregate_and_save_mia(args, ctx)


if __name__ == '__main__':
    main()
