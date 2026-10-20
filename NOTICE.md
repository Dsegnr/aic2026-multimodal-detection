# 来源与第三方依赖

本仓库为参赛实验代码与配置整理，不将第三方检测架构标为自研。新增工具与实验模块从本人的 AIC 工作区导出，具体来源和导出哈希见 `docs/source-manifest.json`。原始工作区和实验资产未被修改。

- [D-FINE](https://github.com/Peterande/D-FINE)：D-FINE 模型、训练框架和基础配置来自该项目；历史配置基于其配置体系编写。使用与修改上游内容应遵守其许可证。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 模型及训练接口，按其适用许可证使用。
- [Co-DETR](https://github.com/Sense-X/Co-DETR)：历史实验涉及该路线，本仓库不分发其完整框架。
- [pycocotools](https://github.com/cocodataset/cocoapi)：COCO 评测对照实现。

模型、数据和上游依赖各自的权利与许可证归原权利人所有。此归档没有为第三方材料重新授权。

清单哈希口径：`source_sha256` 为原工作区文件；`working_copy_sha256` 为导出时本地副本；`published_sha256` 为 Git 存储的文本内容（LF 换行），用于核对公开版本。
