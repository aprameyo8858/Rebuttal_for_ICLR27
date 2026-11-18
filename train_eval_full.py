import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

CODE_VERSION = "full_v3_stable_norm_cw"
USE_TQDM = sys.stderr.isatty()   # no carriage-return spam in redirected log files


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


def atomic_csv_save(df, path):
    tmp_path = f"{path}.tmp.{os.getpid()}.csv"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------------------- model
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
    """Channel normalization (Appendix C.1) as the first layer: inputs live in [0,1] pixel space,
    so the input gradient and the C&W distance are both measured in the same pixel space."""
    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def get_norm_layer(norm_type):
    if norm_type == 'ln':        # per-pixel LN over C
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
    """Zero the gamma of the last norm in every residual block so each block starts as identity.
    (torchvision's zero_init_residual only handles BN-style .weight and breaks with LayerNorm2d.)"""
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
    """Applies transforms on the fly (fresh augmentation every epoch) and returns the sample index."""
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


# ----------------------------------------------------------------------------- C&W
def compute_cw_adv_distance(model, x, y, max_eps=6.0, cw_bs_steps=5, cw_iters=200, cw_lr=0.01, init_const=0.01):
    """
    Untargeted L2 Carlini & Wagner attack in tanh space. x is in [0,1] pixel space.
    Objective: ||x'-x||_2^2 + c * max(Z_y - max_{k!=y} Z_k, 0), 5-step binary search over c.
    Returns (d_adv, success, init_misclassified). Failed/censored samples get d=max_eps, success=False.
    """
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
        w0 = torch.atanh((2.0 * x0 - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))   # finite even for pixels at 0 or 1

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
                    is_adv = logits.argmax(dim=1) != y0     # same forward pass as l2 -> consistent pair
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


def print_cw_attack_statistics(df_eval, dataset, model, seed_label="All Seeds"):
    total = len(df_eval)
    succ = df_eval['cw_success'].astype(bool).values
    init_mis = df_eval['init_misclassified'].astype(bool).values
    attacked = ~init_mis
    flipped = attacked & succ
    failed = ~succ
    valid_d = df_eval['d_adv'].values[succ]
    valid_attacked_d = df_eval['d_adv'].values[flipped]
    print("------------------------------------------------------------------------------------------")
    print(f" C&W Attack Statistics | {dataset.upper()} | {model.upper()} | {seed_label}")
    print(f"   Total Evaluated Samples       : {total}")
    print(f"   Initially Misclassified (d=0) : {init_mis.sum()} ({100.0 * init_mis.mean():.2f}%)")
    print(f"   Correctly Classified Attacked : {attacked.sum()}")
    print(f"   Successfully Flipped by C&W   : {flipped.sum()} (Conditional ASR: {100.0 * flipped.sum() / max(attacked.sum(), 1):.2f}%)")
    print(f"   Failed / Censored (max_eps)   : {failed.sum()} ({100.0 * failed.mean():.2f}%)")
    print(f"   Overall Valid Boundary Rate   : {succ.sum()}/{total} ({100.0 * succ.mean():.2f}%)")
    if len(valid_d) > 0:
        print(f"   Valid d_adv (Mean +/- Std)    : {np.mean(valid_d):.4f} +/- {np.std(valid_d):.4f} (Median: {np.median(valid_d):.4f})")
    if len(valid_attacked_d) > 0:
        print(f"   Flipped-only d_adv (Mean/Med) : {np.mean(valid_attacked_d):.4f} / {np.median(valid_attacked_d):.4f}")
    print("------------------------------------------------------------------------------------------")


def merge_seed_csvs(out_dir, dataset, model, target_seeds):
    eval_dfs, all50k_dfs, missing = [], [], []
    for s in target_seeds:
        p_eval = os.path.join(out_dir, f"full_{dataset}_{model}_seed{s}.csv")
        p_50k = os.path.join(out_dir, f"full_all50k_dynamics_{dataset}_{model}_seed{s}.csv")
        if os.path.exists(p_eval):
            eval_dfs.append(pd.read_csv(p_eval))
        else:
            missing.append(s)
        if os.path.exists(p_50k):
            all50k_dfs.append(pd.read_csv(p_50k))
    if missing:
        print(f"[!] Not merging {dataset}/{model}: missing seed CSVs for seeds {missing} (check logs for failed runs).")
        return
    comb_eval = pd.concat(eval_dfs, ignore_index=True)
    out_eval_csv = os.path.join(out_dir, f"full_{dataset}_{model}_3seeds.csv")
    atomic_csv_save(comb_eval, out_eval_csv)
    print(f"[+] Merged {len(target_seeds)} seeds into: {out_eval_csv} ({len(comb_eval)} rows)")
    print_cw_attack_statistics(comb_eval, dataset, model, seed_label=f"Merged {len(target_seeds)} Seeds")
    if len(all50k_dfs) == len(target_seeds):
        out_50k_csv = os.path.join(out_dir, f"full_all50k_dynamics_{dataset}_{model}_3seeds.csv")
        atomic_csv_save(pd.concat(all50k_dfs, ignore_index=True), out_50k_csv)
        print(f"[+] Merged 50k trajectory dynamics into: {out_50k_csv}")


# ----------------------------------------------------------------------------- training
def lr_at(it, iters_per_epoch, args):
    """Per-iteration linear warmup, then x0.1 at 50% and 75% of training (epochs 100/150 for 200)."""
    warm_iters = args.warmup_epochs * iters_per_epoch
    if warm_iters > 0 and it < warm_iters:
        return args.lr * (it + 1) / warm_iters
    ep = it // iters_per_epoch
    factor = 1.0
    if ep >= int(args.epochs * 0.5):
        factor *= 0.1
    if ep >= int(args.epochs * 0.75):
        factor *= 0.1
    return args.lr * factor


def train_seed(args, seed, train_ds, test_loader, device, num_classes, total_samples, mean, std, std_t):
    set_seed(seed)
    model = build_model(args.model, num_classes, args.norm_layer, mean, std,
                        zero_init_res=not args.no_zero_init_residual).to(device)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, drop_last=False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')

    epoch_losses = np.full((args.epochs, total_samples), np.nan, dtype=np.float32)
    epoch_grads = np.full((args.epochs, total_samples), np.nan, dtype=np.float32)
    iters_per_epoch = len(train_loader)
    chance = 100.0 / num_classes
    it = 0

    for epoch in tqdm(range(args.epochs), desc=f"Train Seed {seed}", disable=not USE_TQDM):
        model.train()
        sum_loss = torch.zeros((), device=device)
        n_correct = torch.zeros((), device=device)
        n_seen = 0
        lr = args.lr
        for idxs, x, y in train_loader:
            lr = lr_at(it, iters_per_epoch, args)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            x = x.to(device, non_blocking=True).requires_grad_(True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            losses = criterion(out, y)
            mean_loss = losses.mean()
            if not torch.isfinite(mean_loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, iter {it}, lr={lr:.4g}. "
                                         f"Lower --lr, increase --warmup-epochs, or try --norm-layer lnchw/gn.")
            optimizer.zero_grad(set_to_none=True)
            mean_loss.backward()

            with torch.no_grad():
                B = x.size(0)
                # LN/GN are strictly per-sample, so B * d(mean loss)/dx_i == d(loss_i)/dx_i exactly.
                g_in = x.grad * B                       # pixel space [0,1] (same space as d_adv)
                if args.grad_space == 'norm':
                    g_in = g_in * std_t                 # dl/dx_norm = dl/dx_pixel * std
                grad_norms = g_in.flatten(1).norm(dim=1)
                idx_np = idxs.numpy()
                epoch_losses[epoch, idx_np] = losses.detach().float().cpu().numpy()
                epoch_grads[epoch, idx_np] = grad_norms.float().cpu().numpy()
                sum_loss += losses.detach().sum()
                n_correct += (out.argmax(dim=1) == y).sum()
                n_seen += B

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # does not affect x.grad above
            optimizer.step()
            it += 1

        if np.isnan(epoch_losses[epoch]).any():
            raise RuntimeError(f"Some samples received no loss in epoch {epoch + 1}.")
        train_loss = sum_loss.item() / n_seen
        train_acc = 100.0 * n_correct.item() / n_seen
        msg = f"[Seed {seed}] Epoch {epoch + 1:3d}/{args.epochs} | lr {lr:.5f} | train loss {train_loss:.4f} | train acc {train_acc:.2f}%"
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            msg += f" | test acc {evaluate_accuracy(model, test_loader, device):.2f}%"
        print(msg, flush=True)

        if args.sanity_epoch > 0 and epoch + 1 == args.sanity_epoch and train_acc < 2.0 * chance:
            raise RuntimeError(f"Train accuracy {train_acc:.2f}% after {epoch + 1} epochs is ~chance ({chance:.1f}%); aborting.")

    return model, epoch_losses, epoch_grads


def load_valid_checkpoint(path, args, min_test_acc):
    if args.force_retrain or not os.path.exists(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    reasons = []
    if ck.get('code_version') != CODE_VERSION:
        reasons.append(f"code_version={ck.get('code_version')}")
    if ck.get('epoch') != args.epochs:
        reasons.append(f"epochs={ck.get('epoch')}")
    if ck.get('norm_layer') != args.norm_layer:
        reasons.append(f"norm_layer={ck.get('norm_layer')}")
    if ck.get('grad_space') != args.grad_space:
        reasons.append(f"grad_space={ck.get('grad_space')}")
    acc = ck.get('test_acc', float('nan'))
    if not np.isfinite(acc) or acc < min_test_acc:
        reasons.append(f"test_acc={acc}")
    if 'state_dict' in ck and not all(torch.isfinite(v).all() for v in ck['state_dict'].values() if v.is_floating_point()):
        reasons.append("non-finite weights")
    if reasons:
        print(f"[!] Ignoring existing checkpoint {path} ({', '.join(reasons)}); retraining.")
        return None
    return ck


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Full-Dataset Training (dynamic aug, per-sample norm) & C&W Evaluation")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--norm-layer', type=str, default='lnchw', choices=['ln', 'lnchw', 'gn', 'bn'],
                        help="'ln' per-pixel LN over C; 'lnchw' LN over (C,H,W)=GroupNorm(1,C); 'gn' GroupNorm(32); 'bn' BatchNorm (couples samples)")
    parser.add_argument('--grad-space', type=str, default='pixel', choices=['pixel', 'norm'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--target-seeds-for-merge', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--grad-clip', type=float, default=5.0, help='Max param-grad norm (0 disables)')
    parser.add_argument('--no-zero-init-residual', action='store_true')
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--sanity-epoch', type=int, default=10, help='Abort if train acc < 2x chance at this epoch (0 disables)')
    parser.add_argument('--min-test-acc', type=float, default=None,
                        help='Abort before C&W if final test acc is below this (default 40 for C10, 20 for C100)')
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--cw-batch-size', type=int, default=512)
    parser.add_argument('--eval-size', type=int, default=50000)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--cw-bs-steps', type=int, default=5)
    parser.add_argument('--cw-iters', type=int, default=200)
    parser.add_argument('--cw-lr', type=float, default=0.01)
    parser.add_argument('--cw-init-const', type=float, default=0.01)
    parser.add_argument('--max-eps', type=float, default=6.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.merge_only:
        merge_seed_csvs(args.out_dir, args.dataset, args.model, args.target_seeds_for_merge)
        return

    if args.norm_layer == 'bn':
        print("[!] WARNING: BatchNorm couples samples in a batch; B*x.grad is NOT the per-sample input gradient.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    mean, std, num_classes, ds_cls = get_dataset_stats(args.dataset)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    L_u = np.log(num_classes) + 1e-3
    min_test_acc = args.min_test_acc if args.min_test_acc is not None else (40.0 if num_classes == 10 else 20.0)

    ckpt_base = os.path.join(args.out_dir, "checkpoints_full", f"{args.dataset}_{args.model}_{args.norm_layer}")
    os.makedirs(ckpt_base, exist_ok=True)

    # Normalization lives inside the model -> datasets yield [0,1] tensors.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_eval = transforms.ToTensor()

    raw_train = ds_cls(root=args.data_dir, train=True, download=True)
    raw_test = ds_cls(root=args.data_dir, train=False, download=True)
    total_samples = len(raw_train)
    targets_all = np.array(raw_train.targets, dtype=np.int64)

    if args.eval_size <= 0 or args.eval_size >= total_samples:
        eval_indices = np.arange(total_samples, dtype=np.int64)
    else:
        rng = np.random.RandomState(args.split_seed)
        eval_indices = np.sort(rng.choice(total_samples, size=args.eval_size, replace=False))
    eval_idx_file = os.path.join(args.out_dir, f"eval_indices_{args.dataset}_n{len(eval_indices)}.npy")
    if not os.path.exists(eval_idx_file):
        atomic_npy_save(eval_indices, eval_idx_file)

    train_ds = IndexedTransformedDataset(raw_train, transform_train)
    eval_ds = IndexedTransformedDataset(raw_train, transform_eval)
    test_ds = IndexedTransformedDataset(raw_test, transform_eval)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for seed in args.seeds:
        print(f"\n[*] [Full 50k] {args.model} ({args.norm_layer.upper()}) | {args.dataset} | Seed {seed} | "
              f"lr {args.lr} warmup {args.warmup_epochs} clip {args.grad_clip} | C&W subset {len(eval_indices)}")
        final_ckpt_path = os.path.join(ckpt_base, f"target_seed{seed}_final.pth")
        ck = load_valid_checkpoint(final_ckpt_path, args, min_test_acc)

        if ck is not None:
            print(f"[*] Loaded verified checkpoint (test acc {ck['test_acc']:.2f}%). Skipping training.")
            model = build_model(args.model, num_classes, args.norm_layer, mean, std,
                                zero_init_res=not args.no_zero_init_residual).to(device)
            model.load_state_dict(ck['state_dict'])
            epoch_losses, epoch_grads, test_acc = ck['epoch_losses'], ck['epoch_grads'], ck['test_acc']
            d_adv_all = ck['d_adv_all_cached'].copy()
            cw_success_all = ck['cw_success_all_cached'].copy()
            init_mis_all = ck['init_mis_all_cached'].copy()
        else:
            model, epoch_losses, epoch_grads = train_seed(args, seed, train_ds, test_loader, device,
                                                          num_classes, total_samples, mean, std, std_t)
            test_acc = evaluate_accuracy(model, test_loader, device)
            print(f"[*] Seed {seed} finished {args.epochs} epochs | Clean Test Accuracy: {test_acc:.2f}%")
            d_adv_all = np.full(total_samples, np.nan, dtype=np.float64)
            cw_success_all = np.zeros(total_samples, dtype=bool)
            init_mis_all = np.zeros(total_samples, dtype=bool)
            if not np.isfinite(test_acc) or test_acc < min_test_acc:
                atomic_torch_save({'state_dict': model.state_dict(), 'test_acc': test_acc,
                                   'epoch_losses': epoch_losses, 'epoch_grads': epoch_grads},
                                  final_ckpt_path.replace('.pth', '_FAILED.pth'))
                raise RuntimeError(f"Test accuracy {test_acc} < {min_test_acc}; not running C&W on a broken model.")

        stats = derive_stats(epoch_losses, epoch_grads, L_u, args.epochs)

        def save_ckpt():
            atomic_torch_save({
                'code_version': CODE_VERSION, 'epoch': args.epochs, 'seed': seed, 'model': args.model,
                'dataset': args.dataset, 'norm_layer': args.norm_layer, 'grad_space': args.grad_space,
                'test_acc': test_acc, 'state_dict': model.state_dict(),
                'epoch_losses': epoch_losses, 'epoch_grads': epoch_grads,
                'd_adv_all_cached': d_adv_all, 'cw_success_all_cached': cw_success_all,
                'init_mis_all_cached': init_mis_all,
            }, final_ckpt_path)

        if ck is None:
            save_ckpt()   # persist training before the (long) C&W stage

        to_attack = eval_indices[np.isnan(d_adv_all[eval_indices])]
        if len(to_attack) > 0:
            print(f"[*] Running C&W on {len(to_attack)} samples (Seed {seed}, batch {args.cw_batch_size})...")
            loader = DataLoader(Subset(eval_ds, to_attack.tolist()), batch_size=args.cw_batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
            for b, (idxs, x, y) in enumerate(tqdm(loader, desc=f"C&W Seed {seed}", disable=not USE_TQDM)):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                bd, bs, bm = compute_cw_adv_distance(model, x, y, max_eps=args.max_eps, cw_bs_steps=args.cw_bs_steps,
                                                     cw_iters=args.cw_iters, cw_lr=args.cw_lr,
                                                     init_const=args.cw_init_const)
                idx_np = idxs.numpy()
                d_adv_all[idx_np], cw_success_all[idx_np], init_mis_all[idx_np] = bd, bs, bm
                print(f"    C&W batch {b + 1}/{len(loader)} done", flush=True)
            save_ckpt()

        common = {'model': args.model, 'dataset': args.dataset, 'norm_layer': args.norm_layer, 'seed': seed}
        cols = ['csl', 'csg', 'final_loss', 'final_grad', 'csl_diff_40', 'csl_diff_50',
                'lt_mean', 'lt_std', 'lt_iqr']
        df_eval = pd.DataFrame({
            **common, 'eval_idx': np.arange(len(eval_indices)), 'global_idx': eval_indices,
            'target': targets_all[eval_indices],
            **{c: stats[c][eval_indices] for c in cols},
            'd_adv': d_adv_all[eval_indices],
            'd_adv_proxy': stats['d_adv_proxy'][eval_indices],
            'd_adv_1_proxy': stats['d_adv_1_proxy'][eval_indices],
            'init_misclassified': init_mis_all[eval_indices],
            'cw_success': cw_success_all[eval_indices],
        })
        print_cw_attack_statistics(df_eval, args.dataset, args.model, seed_label=f"Seed {seed}")
        seed_eval_csv = os.path.join(args.out_dir, f"full_{args.dataset}_{args.model}_seed{seed}.csv")
        atomic_csv_save(df_eval, seed_eval_csv)

        df_50k = pd.DataFrame({
            **common, 'global_idx': np.arange(total_samples), 'target': targets_all,
            **{c: stats[c] for c in cols},
            'd_adv': d_adv_all, 'd_adv_proxy': stats['d_adv_proxy'], 'd_adv_1_proxy': stats['d_adv_1_proxy'],
            'init_misclassified': init_mis_all, 'cw_success': cw_success_all,
        })
        seed_50k_csv = os.path.join(args.out_dir, f"full_all50k_dynamics_{args.dataset}_{args.model}_seed{seed}.csv")
        atomic_csv_save(df_50k, seed_50k_csv)
        print(f"[+] Saved seed {seed} CSVs to {seed_eval_csv} and {seed_50k_csv}")

    if len(args.seeds) > 1:
        merge_seed_csvs(args.out_dir, args.dataset, args.model, args.target_seeds_for_merge)


if __name__ == '__main__':
    main()
