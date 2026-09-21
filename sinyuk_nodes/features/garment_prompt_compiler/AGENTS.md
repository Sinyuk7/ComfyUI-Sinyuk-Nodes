`sinyuk_nodes/features/garment_prompt_compiler/` 实现一个两阶段的服装换装提示词流程：

```text
Image 1 人物图
+ Images 2..N 目标穿搭参考图
        ↓
Vision LLM
(system_prompt.txt + user_prompt.txt + schema.json)
        ↓
GarmentAnalysis JSON
        ↓
compiler.py
        ↓
garment_replacement.txt + outfit_item.txt
        ↓
最终 Image Editing Prompt
        ↓
Nano Banana / GPT Image
```

核心职责：

* `system_prompt.txt`：定义 Vision LLM 的分析规则和图片角色。
* `user_prompt.txt`：触发当前图片分析任务，保持简洁。
* `schema.json`：定义 `GarmentAnalysis JSON` 的固定数据结构。
* `compiler.py`：纯 Python 编译器，把 JSON 确定性组装成最终 Prompt，不调用 LLM、不重新推理。
* `outfit_item.txt`：单个服装 / 鞋 / 包 / 配件的 Prompt 模板。
* `garment_replacement.txt`：最终换装 Prompt 的全局模板。
* `__init__.py`：Feature 的导出和注册入口。

核心边界：

```text
Vision LLM = 理解图片
Schema = 结构化数据协议
Compiler = 确定性组装
Templates = Prompt 表达结构
Image Model = 执行最终换装
```

Image 1 只提供人物、姿势、构图和视角；Images 2..N 定义完整目标 outfit，包括服装、鞋、包、腰带等。

Reference 应遵循“最小充分引用”：整体版型使用必要的 main reference，具体局部细节使用最清晰、最权威的 detail reference，避免无意义的多图交叉引用。
