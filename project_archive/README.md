# 项目归档

这个目录集中保存当前 TEST 与 APP 运行链路之外的旧代码、历史文档和运行产物。文件已移动，未删除。

| 目录 | 原位置 | 内容与用途 |
| --- | --- | --- |
| `vscode-extension/` | 项目根目录的 `src/`、`media/`、Node/TypeScript 配置和旧 `.vscode` 扩展设置 | VS Code Hello World 扩展模板；其 `node_modules/` 是可重新安装的本地依赖 |
| `early-prototype/` | 根目录 `acquisition/`、`processing/` 和 `app/web/` | 旧 STM32 采集骨架、同步器、配置和相机页面；保留作设计参考 |
| `history/` | `docs/history/` | 整理前的 README、旧运行指南与上一次归档说明 |
| `logs/test/` | `test/logs/` | TEST 实验的历史采集日志；部分文件在整理前已有 Git 改动 |
| `python-cache/` | `app/` 与 `test/` 下的 `__pycache__/` | Python 自动生成的缓存；再次运行代码时会重新生成 |

运行 TEST 时，`test/logs/` 与 `__pycache__/` 可能重新出现。正式 APP 的运行日志仍写入项目根目录 `logs/`，没有移动。

历史文档中出现的旧文件路径反映写作时的目录结构。当前项目入口和进度以根目录 [`README.md`](../README.md) 与 [`docs/09_项目现状与目录说明.md`](../docs/09_项目现状与目录说明.md) 为准。
