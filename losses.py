"""
지식 증류 Loss 함수

1. VanillaKDLoss: Hinton et al. (2015) — soft label + hard label
2. MGDLoss: Yang et al. (ECCV 2022) — feature map 랜덤 마스킹 + generation block으로 teacher feature 복원

2-a 단계에서 MGDLoss의 _generate_mask()를 attribution-guided로 교체할 예정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaKDLoss(nn.Module):
    """
    Vanilla KD: L = alpha * KL(soft_s || soft_t) * T^2 + (1-alpha) * CE(s, y)
    """
    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits_student, logits_teacher, labels):
        loss_ce = self.ce_loss(logits_student, labels)
        soft_s = F.log_softmax(logits_student / self.temperature, dim=1)
        soft_t = F.softmax(logits_teacher / self.temperature, dim=1)
        loss_kd = F.kl_div(soft_s, soft_t, reduction='batchmean') * (self.temperature ** 2)
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
        return loss, {'loss_total': loss.item(), 'loss_ce': loss_ce.item(), 'loss_kd': loss_kd.item()}


class MGDLoss(nn.Module):
    """
    Masked Generative Distillation (ECCV 2022)

    Student feature map → 랜덤 마스킹 → generation block → teacher feature 복원
    L = alpha * L_KD + (1-alpha) * L_CE + beta * MSE(generated, teacher_feat)

    Args:
        student_channels: Student feature map 채널 수
        teacher_channels: Teacher feature map 채널 수
        mask_ratio: 마스킹 비율 (0~1, 기본 0.5)
        beta: MGD loss 가중치
    """
    def __init__(self, student_channels, teacher_channels,
                 mask_ratio=0.5, temperature=4.0, alpha=0.9, beta=7e-3):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()

        # Channel alignment (student → teacher 채널 맞춤)
        self.align = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)

        # Generation block: 마스킹된 feature → teacher feature 복원
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _generate_mask(self, feature):
        """
        랜덤 binary mask 생성 — 2-a 단계에서 이 메소드를 교체할 예정

        Args:
            feature: (B, C, H, W)
        Returns:
            mask: (B, 1, H, W), 0=마스킹, 1=유지
        """
        B, C, H, W = feature.shape
        num_pixels = H * W
        num_mask = int(num_pixels * self.mask_ratio)

        mask = torch.ones(B, 1, H, W, device=feature.device)
        for i in range(B):
            indices = torch.randperm(num_pixels, device=feature.device)[:num_mask]
            mask[i].view(-1)[indices] = 0
        return mask

    def forward(self, logits_student, logits_teacher,
                feature_student, feature_teacher, labels):
        """
        Args:
            logits_student/teacher: (B, num_classes)
            feature_student: (B, C_s, H, W)
            feature_teacher: (B, C_t, H, W)
            labels: (B,)
        """
        # CE loss
        loss_ce = self.ce_loss(logits_student, labels)

        # KD loss
        soft_s = F.log_softmax(logits_student / self.temperature, dim=1)
        soft_t = F.softmax(logits_teacher / self.temperature, dim=1)
        loss_kd = F.kl_div(soft_s, soft_t, reduction='batchmean') * (self.temperature ** 2)

        # MGD loss
        feat_s_aligned = self.align(feature_student)       # channel alignment
        mask = self._generate_mask(feat_s_aligned)          # 랜덤 마스크
        feat_s_masked = feat_s_aligned * mask               # 마스킹 적용
        feat_s_generated = self.generation(feat_s_masked)   # teacher feature 복원
        loss_mgd = F.mse_loss(feat_s_generated, feature_teacher.detach())

        # Total loss
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce + self.beta * loss_mgd

        return loss, {
            'loss_total': loss.item(), 'loss_ce': loss_ce.item(),
            'loss_kd': loss_kd.item(), 'loss_mgd': loss_mgd.item()
        }
