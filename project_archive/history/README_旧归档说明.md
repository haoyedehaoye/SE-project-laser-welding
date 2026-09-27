# 归档代码

这里保留不在当前 TEST 或 APP 运行链路上的旧实现，便于查阅。没有删除原始代码。

| 当前目录 | 原位置 | 原因 |
| --- | --- | --- |
| `early-prototype/acquisition/` | `acquisition/` | 尚未贯通的早期 STM32 采集骨架，协议与正式 APP 不同 |
| `early-prototype/processing/` | `processing/` | 早期同步器和配置，未被正式 APP 导入 |
| `vscode-extension/` | 根目录的 `src/`、`media/`、Node/TypeScript 配置和扩展开发设置 | VS Code Hello World 模板，与 Python 数据平台无调用关系 |

`vscode-extension/node_modules/` 是旧扩展的本地依赖，可重新安装。旧扩展的启动配置也一同归档；若未来恢复扩展开发，需要重新校正路径与命令注册。

历史文档中的 `acquisition/` 和 `processing/` 路径指整理前的位置。正式系统入口仍为项目根目录的 `app/`。
