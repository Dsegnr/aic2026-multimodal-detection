# 环境与复现

## 1. 依赖

训练使用 PyTorch，YOLO 分支使用 Ultralytics；D-FINE 分支需另行克隆 [D-FINE](https://github.com/Peterande/D-FINE)。`requirements.txt` 列出直接依赖，不是比赛时环境的锁文件。

## 2. 数据

自行准备合法取得的比赛数据与权重。YOLO 配置默认使用 `data/` 下的数据；按实际目录调整配置。评测真值采用逐图 TXT：`class cx cy w h`；预测采用 `class cx cy w h confidence`，坐标为归一化坐标。COCO 配置中的标签和图片路径也需对应修改。

## 3. YOLO 基线

```bash
python scripts/train.py --model weights/yolov8s.pt --data configs/aic_lbproxy.yaml --epochs 100 --imgsz 640 --batch 16
```

## 4. D-FINE

将 `configs/archive/dfine_custom/` 中选用的配置复制到 D-FINE 的 `configs/dfine/custom/`，数据集配置放入其 `configs/dataset/`。配置的 `__include__` 依赖该上游仓库的 runtime、dataloader、optimizer 和模型配置；需要保留上游目录结构。

设置 `DFINE_ROOT` 指向克隆的上游目录，并准备 12 类模型对应权重。训练入口为 `scripts/dfine_run.py`，参数透传上游 `train.py`。推理时 `--config` 使用配置的绝对路径：

```bash
python scripts/dfine_infer.py --config /path/to/D-FINE/configs/dfine/custom/your_config.yml --weights weights/your_checkpoint.pth --list splits/test.txt --out preds/test --imgsz 1280 --scales 1.0,1.0,0.83 --flips none,lr,none
```

`splits/test.txt` 每行一个图像路径。多模态推理额外指定 `--ir`、`--dp`；不要依赖历史默认路径。多模态分支涉及额外上游集成，历史配置不等同于一条可无准备运行的端到端命令。

## 5. 评测与输出检查

```bash
python src/eval/map_eval.py --gt-dir path/to/ground_truth --pred-dir preds/test --json-out metrics.json
python scripts/check_submission.py --pred-dir preds/test --test-dir path/to/test_images
```

本地指标用于实验分析。最终奖项以赛事结果为准，不把本地验证分数视为线上榜分。此次发布验证限于代码静态检查、已有评测单元测试和提交格式检查；没有重新训练模型或重新提交赛事。
