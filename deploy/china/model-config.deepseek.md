# 中国实例模型配置模板 —— 全 DeepSeek（美国生产不动）

> 2026-09-08 决策：中美两台服务器全部角色只允许中国模型。
> 主模型 **deepseek-v4-flash**；视觉模型 **deepseek-v4-flash-vision-exp**。
> 美国生产环境暂保持 openrouter/gpt-5.6-terra（切换时复用本模板即可）。
> Embedding 已是中国：siliconflow / BAAI/bge-m3（`[EMBEDDING]`，无需改）。

## 1. config.ini 需写入的段落

```ini
[MAIN]
provider_name = deepseek
provider_model = deepseek-v4-flash

; 角色覆盖层:所有角色全切 DeepSeek(长任务/解释/流式/视觉读 [MODEL])
[MODEL]
chat_provider = deepseek
chat_model = deepseek-v4-flash
interpret_provider = deepseek
interpret_model = deepseek-v4-flash
stream_provider = deepseek
stream_model = deepseek-v4-flash
vision_provider = deepseek
vision_model = deepseek-v4-flash-vision-exp

; 长任务视觉兜底(默认 minimax/MiniMax-M3 已是中国,此处统一到 deepseek vision)
[LONG_TASK]
vision_provider = deepseek
vision_model = deepseek-v4-flash-vision-exp
```

## 2. .env 必设项

```bash
DEEPSEEK_API_KEY=...            # 生产 .env 已有,同 key 复制
REACT_INTERPRET_PROVIDER=deepseek   # 默认 openrouter!!必覆写(technical_interpretation.py)
REACT_INTERPRET_MODEL=deepseek-v4-flash
```

## 3. 全模型调用点盘点(2026-09-08 代码审计)

| 调用点 | 默认值 | 切后状态 |
|---|---|---|
| [MAIN] 主对话/ReAct | openrouter gpt-5.6-terra(生产现状) | → deepseek-v4-flash |
| [MODEL] chat/interpret/stream/vision | 空→跟随 MAIN | → 全 deepseek |
| long_task/config.py provider_family | **deepseek**(原生) | ✅ 无需动 |
| long_task/config.py vision | minimax/MiniMax-M3(中国) | 可选统一 → 模板已设 |
| technical_interpretation.py | **openrouter** / openai/gpt-5.6-terra | ⚠️ 必须 env 覆写(见上) |
| grounded_interpretation.py | deepseek-v4-flash(原生) | ✅ 无需动 |
| react_tools REACT_SCORE | deepseek-v4-flash(原生) | ✅ 无需动 |
| knowledge.py EMBEDDING_CHAT | 默认 gpt-3.5-turbo,被 [MAIN] 覆写 | ⚠️ 注意:该调用走 SiliconFlow 的 OpenAI 兼容客户端,若 SiliconFlow 不提供 `deepseek-v4-flash` 模型名,需显式设 `EMBEDDING_CHAT_MODEL` 为 SiliconFlow 货架上的模型(如 Qwen),或确认该路径生产未触发 |
| Embedding 向量 | siliconflow / BAAI/bge-m3 | ✅ 中国 |

## 4. 冒烟清单(切后必须验证)

1. 文字专利提问一条(阶梯/语义重排/落库正常)
2. **图片/外观比对上传一条**(vision=deepseek-v4-flash-vision-exp 生效;卖家线功能不能回归)
3. 知识库问答一条(EMBEDDING_CHAT 路径是否可用)
4. 检索"架构级理解"(technical_interpretation)一条(确认走 deepseek 而非 openrouter——日志或 env 生效)

## 5. 回滚

纯配置改动:config.ini [MAIN]/[MODEL] 改回 + 删除 env 覆写 → 重启 backend+celery 即还原。
