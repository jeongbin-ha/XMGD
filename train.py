"""
XMGD 프로젝트 - 1단계: MGD 구현 및 Vanilla KD 비교

사용법:
  python train.py --mode all --epochs 200 --gpu 0     # 전체 실험
  python train.py --mode teacher --epochs 200          # Teacher만
  python train.py --mode mgd --epochs 200              # MGD만
  python train.py --mode all --epochs 5                # 동작 확인용
"""

import os, time, argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models import resnet56, resnet20, count_parameters
from losses import VanillaKDLoss, MGDLoss


class Config:
    num_classes = 100
    data_dir = './data'
    epochs = 200
    batch_size = 128
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    lr_milestones = [100, 150]
    lr_gamma = 0.1
    kd_temperature = 4.0
    kd_alpha = 0.9
    mgd_mask_ratio = 0.5
    mgd_beta = 7e-3
    checkpoint_dir = './checkpoints'
    log_dir = './logs'


def get_dataloaders(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = torchvision.datasets.CIFAR100(config.data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(config.data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ===== 학습 함수 =====
def train_one_epoch_ce(model, loader, optimizer, criterion, device):
    """CE loss만으로 한 epoch 학습 (Teacher / Scratch 용)"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += logits.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total


def train_one_epoch_kd(student, teacher, loader, loss_fn, optimizer, device):
    """KD 기반 한 epoch 학습 (Vanilla KD 용)"""
    student.train(); teacher.eval()
    total_loss, correct, total = 0, 0, 0
    info_sum = {}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits_t = teacher(images)
        logits_s = student(images)
        loss, info = loss_fn(logits_s, logits_t, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * images.size(0)
        for k, v in info.items(): info_sum[k] = info_sum.get(k, 0) + v * images.size(0)
        correct += logits_s.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total, {k: v/total for k, v in info_sum.items()}


def train_one_epoch_mgd(student, teacher, loader, loss_fn, optimizer, device):
    """MGD 한 epoch 학습"""
    student.train(); teacher.eval()
    total_loss, correct, total = 0, 0, 0
    info_sum = {}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits_t, feat_t = teacher(images, return_feature=True)
        logits_s, feat_s = student(images, return_feature=True)
        loss, info = loss_fn(logits_s, logits_t, feat_s, feat_t, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * images.size(0)
        for k, v in info.items(): info_sum[k] = info_sum.get(k, 0) + v * images.size(0)
        correct += logits_s.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total, {k: v/total for k, v in info_sum.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, correct5, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += logits.max(1)[1].eq(labels).sum().item()
        correct5 += logits.topk(5, dim=1)[1].eq(labels.unsqueeze(1)).any(1).sum().item()
        total += labels.size(0)
    return 100. * correct / total, 100. * correct5 / total


# ===== 실험 실행 =====
def run_experiment(mode, config, device, train_loader, test_loader, teacher=None):
    """통합 실험 실행 함수"""

    if mode == 'teacher':
        print("=" * 60)
        print("Teacher (ResNet-56) 학습")
        model = resnet56(config.num_classes).to(device)
        print(f"파라미터: {count_parameters(model):,}")
        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.lr_milestones, config.lr_gamma)
        criterion = nn.CrossEntropyLoss()
        train_fn = lambda: train_one_epoch_ce(model, train_loader, optimizer, criterion, device)
        log_extra = lambda info: ""

    elif mode == 'scratch':
        print("=" * 60)
        print("Student Scratch (ResNet-20) 학습")
        model = resnet20(config.num_classes).to(device)
        print(f"파라미터: {count_parameters(model):,}")
        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.lr_milestones, config.lr_gamma)
        criterion = nn.CrossEntropyLoss()
        train_fn = lambda: train_one_epoch_ce(model, train_loader, optimizer, criterion, device)
        log_extra = lambda info: ""

    elif mode == 'vanilla_kd':
        print("=" * 60)
        print("Vanilla KD (ResNet-56 → ResNet-20)")
        model = resnet20(config.num_classes).to(device)
        loss_fn = VanillaKDLoss(config.kd_temperature, config.kd_alpha)
        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.lr_milestones, config.lr_gamma)
        train_fn = lambda: train_one_epoch_kd(model, teacher, train_loader, loss_fn, optimizer, device)
        log_extra = lambda info: f" CE:{info.get('loss_ce',0):.3f} KD:{info.get('loss_kd',0):.3f}"

    elif mode == 'mgd':
        print("=" * 60)
        print("MGD (ResNet-56 → ResNet-20)")
        model = resnet20(config.num_classes).to(device)
        loss_fn = MGDLoss(64, 64, config.mgd_mask_ratio,
                          config.kd_temperature, config.kd_alpha, config.mgd_beta).to(device)
        # MGD generation block 파라미터도 optimizer에 포함
        params = list(model.parameters()) + list(loss_fn.parameters())
        optimizer = optim.SGD(params, lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.lr_milestones, config.lr_gamma)
        train_fn = lambda: train_one_epoch_mgd(model, teacher, train_loader, loss_fn, optimizer, device)
        log_extra = lambda info: f" CE:{info.get('loss_ce',0):.3f} KD:{info.get('loss_kd',0):.3f} MGD:{info.get('loss_mgd',0):.3f}"

    best_acc = 0
    log = []

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        result = train_fn()

        # train_fn returns either (loss, acc) or (loss, acc, info)
        if len(result) == 2:
            train_loss, train_acc = result
            info = {}
        else:
            train_loss, train_acc, info = result

        test_top1, test_top5 = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        log.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                     'test_top1': test_top1, 'test_top5': test_top5, **info})

        if epoch % 10 == 0 or epoch <= 2 or epoch == config.epochs:
            extra = log_extra(info)
            print(f"  [Epoch {epoch:3d}/{config.epochs}]{extra} | "
                  f"Train: {train_acc:.1f}% | Test: {test_top1:.2f}% | {elapsed:.1f}s")

        if test_top1 > best_acc:
            best_acc = test_top1
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f'{mode}_best.pth'))

    print(f"  >>> Best: {best_acc:.2f}%\n")

    os.makedirs(config.log_dir, exist_ok=True)
    with open(os.path.join(config.log_dir, f'{mode}_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    return best_acc, model


def main():
    parser = argparse.ArgumentParser(description='XMGD Project - Stage 1')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['teacher', 'scratch', 'vanilla_kd', 'mgd', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--mgd_beta', type=float, default=7e-3)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    config = Config()
    for k in ['epochs', 'batch_size', 'lr']:
        setattr(config, k, getattr(args, k))
    config.mgd_mask_ratio = args.mask_ratio
    config.mgd_beta = args.mgd_beta
    config.kd_temperature = args.temperature

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print()

    train_loader, test_loader = get_dataloaders(config)
    results = {}
    teacher_model = None

    modes = ['teacher', 'scratch', 'vanilla_kd', 'mgd'] if args.mode == 'all' else [args.mode]

    for mode in modes:
        # KD/MGD는 teacher 필요
        if mode in ['vanilla_kd', 'mgd'] and teacher_model is None:
            teacher_path = os.path.join(config.checkpoint_dir, 'teacher_best.pth')
            if not os.path.exists(teacher_path):
                print(f"Teacher 체크포인트가 없습니다. 먼저 --mode teacher로 학습하세요.")
                return
            teacher_model = resnet56(config.num_classes).to(device)
            teacher_model.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
            teacher_model.eval()
            t_acc, _ = evaluate(teacher_model, test_loader, device)
            print(f"  Loaded Teacher: {t_acc:.2f}%\n")

        acc, model = run_experiment(mode, config, device, train_loader, test_loader, teacher_model)
        results[mode] = acc

        if mode == 'teacher':
            teacher_model = model
            teacher_model.eval()

    # 결과 요약
    if len(results) > 1:
        print("=" * 60)
        print("            실험 결과 요약")
        print("=" * 60)
        name_map = {'teacher': 'Teacher (ResNet-56)', 'scratch': 'Student Scratch',
                    'vanilla_kd': 'Vanilla KD', 'mgd': 'MGD'}
        for mode, acc in results.items():
            print(f"  {name_map.get(mode, mode):<25} {acc:.2f}%")
        print("=" * 60)
        if 'scratch' in results and 'mgd' in results:
            print(f"  MGD vs Scratch:     +{results['mgd'] - results['scratch']:.2f}%")
        if 'vanilla_kd' in results and 'mgd' in results:
            g = results['mgd'] - results['vanilla_kd']
            print(f"  MGD vs Vanilla KD:  {'+' if g>=0 else ''}{g:.2f}%")


if __name__ == "__main__":
    main()
