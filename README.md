# Orbit Wars PPO Mission Selector v2

Pipeline đúng theo thiết kế:

```text
obs
 ↓
lb-1200 build WorldModel / parse observation
 ↓
lb-1200 generate missions
 ↓
lấy top K missions
 ↓
PPO chọn 1 trong top K mission hoặc STOP
 ↓
lb-1200 execute mission được chọn thành moves
 ↓
env.step(moves)
 ↓
tính reward
 ↓
PPO update / self-play
```

## Cấu trúc thư mục

```text
orbit_ppo_mission_v2/
├── main.py                         # file top-level để submit Kaggle
├── requirements.txt
├── README.md
├── checkpoints/
│   ├── bc_policy.pt                # sinh ra sau behavior cloning
│   └── ppo_selfplay.pt             # sinh ra sau PPO fine-tune
├── data/
│   └── bc_dataset.npz              # dataset imitation từ lb-1200
├── scripts/
│   └── make_submission.sh
└── src/
    └── orbit_rl/
        ├── lb1200_strategy.py      # code lb-1200 đã patch mission_selector hook
        ├── mission_interface.py    # generate_top_missions, infer_teacher_action
        ├── mission_encoder.py      # encode world + top-K missions
        ├── topk_policy.py          # Actor-Critic chọn STOP/top-K mission
        ├── bc_dataset.py           # thu data từ teacher lb-1200
        ├── train_bc.py             # Behavior Cloning
        ├── ppo_buffer.py
        ├── ppo_update.py
        ├── ppo_mission_agent.py    # lb-1200 + PPO selector
        ├── selfplay_env.py
        ├── train_ppo_selfplay.py   # PPO fine-tune bằng teacher/self-play
        ├── reward.py
        └── inference_agent.py      # load checkpoint, fallback lb-1200
```

## Cài đặt

```bash
cd orbit_ppo_mission_v2
pip install -r requirements.txt
```

## Bước 1: tạo dataset Behavior Cloning từ lb-1200

```bash
python -m src.orbit_rl.bc_dataset \
  --episodes 100 \
  --top-k 8 \
  --save data/bc_dataset.npz
```

Dataset lưu các mẫu:

```text
input  = global_features + top-K mission features + mask
label  = action của teacher lb-1200
         0 = STOP
         1..K = mission rank action-1
```

## Bước 2: train Behavior Cloning

```bash
python -m src.orbit_rl.train_bc \
  --data data/bc_dataset.npz \
  --epochs 20 \
  --top-k 8 \
  --save checkpoints/bc_policy.pt
```

## Bước 3: PPO fine-tune bằng lb-1200/self-play

Ban đầu nên để self-play start muộn để agent không drift quá sớm:

```bash
python -m src.orbit_rl.train_ppo_selfplay \
  --init checkpoints/bc_policy.pt \
  --episodes 500 \
  --top-k 8 \
  --selfplay-start 200 \
  --save checkpoints/ppo_selfplay.pt
```

## Bước 4: đóng gói submit

Phải để `main.py` ở root level của archive.

```bash
bash scripts/make_submission.sh $(pwd) /kaggle/working/submission.zip
```

Kiểm tra đúng:

```bash
unzip -l /kaggle/working/submission.zip | head -40
```

Phải thấy:

```text
main.py
src/orbit_rl/...
checkpoints/ppo_selfplay.pt
```

Không được thấy:

```text
orbit_ppo_mission_v2/main.py
```

## Ghi chú an toàn

`main.py` không dùng `__file__`, vì Kaggle có thể exec code khiến `__file__` không tồn tại.

`inference_agent.py` sẽ ưu tiên load:

1. `checkpoints/ppo_selfplay.pt`
2. `checkpoints/bc_policy.pt`
3. fallback về lb-1200 heuristic nếu model lỗi

Do đó submission không nên crash nếu checkpoint bị thiếu hoặc lỗi.
