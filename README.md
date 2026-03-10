# XMGD Project - Stage 1: MGD 구현 및 비교 실험

## 프로젝트 구조
```
xmgd_project/
├── models.py      # CIFAR용 ResNet (Teacher: ResNet-56, Student: ResNet-20)
├── losses.py      # Loss 함수 (Vanilla KD, MGD)
├── train.py       # 메인 학습 스크립트
├── checkpoints/   # 모델 체크포인트 (자동 생성)
├── logs/          # 학습 로그 JSON (자동 생성)
└── data/          # CIFAR-100 (자동 다운로드)
```

## 환경 설정
```bash
pip install torch torchvision
```

## 실행 방법

### 전체 실험 한 번에
```bash
python train.py --mode all --epochs 200 --gpu 0
```
실행 순서: Teacher → Student Scratch → Vanilla KD → MGD

### 단계별 실행
```bash
python train.py --mode teacher --epochs 200      # 1) Teacher 학습
python train.py --mode scratch --epochs 200      # 2) Student 단독
python train.py --mode vanilla_kd --epochs 200   # 3) Vanilla KD
python train.py --mode mgd --epochs 200          # 4) MGD
```

### 동작 확인용 (빠른 테스트)
```bash
python train.py --mode all --epochs 5
```

## 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| --epochs | 200 | 학습 에폭 수 |
| --batch_size | 128 | 배치 크기 |
| --lr | 0.1 | 학습률 (100, 150 epoch에서 x0.1) |
| --temperature | 4.0 | KD 소프트맥스 온도 |
| --mask_ratio | 0.5 | MGD 마스킹 비율 |
| --mgd_beta | 7e-3 | MGD loss 가중치 |

## 기대 결과 (CIFAR-100, 200 epochs)
| 방법 | 예상 Top-1 (%) |
|------|---------------|
| Teacher (ResNet-56) | ~72 |
| Student Scratch (ResNet-20) | ~69 |
| Vanilla KD | ~70.5 |
| MGD | ~71.5 |

## 코드 구조 핵심 포인트

### models.py
- `forward(x, return_feature=True)` → logits + feature map 동시 반환
- Feature map shape: (B, 64, 8, 8) — 마지막 residual group 출력

### losses.py
- `MGDLoss._generate_mask()` → **이 메소드가 2-a 단계에서 교체 대상**
- 현재: 랜덤 마스킹 / 향후: attribution-guided 마스킹
- MGD generation block 파라미터가 optimizer에 포함되어야 함

### train.py
- `--mode all`로 전체 실험을 순차 실행
- 학습 로그가 JSON으로 저장 → 이후 비교 분석에 활용

## 다음 단계 (2-a)
losses.py에서 `_generate_mask()`를 attribution-guided로 교체:
1. pytorch-grad-cam으로 teacher attribution 사전 계산
2. Attribution 값 기반 확률적 마스킹 구현
3. MGD(랜덤) vs XMGD(attribution-guided) 비교
