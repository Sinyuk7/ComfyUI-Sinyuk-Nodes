`sinyuk_nodes/features/prompt_builder/` 实现一个两阶段的服装换装提示词流程：

```text
Image 1 人物图
+ Images 2..N 目标穿搭参考图
        ↓
Vision LLM
(system.txt + user.txt + schema.json)
        ↓
GarmentAnalysis JSON
        ↓
compiler.py
        ↓
template.txt + templates/outfit_item.txt
        ↓
最终 Image Editing Prompt
        ↓
Nano Banana / GPT Image
```

核心职责：

* `system.txt`：定义 Vision LLM 的分析规则和图片角色。
* `user.txt`：触发当前图片分析任务，保持简洁。
* `schema.json`：定义 `GarmentAnalysis JSON` 的固定数据结构。
* `compiler.py`：纯 Python 编译器，把 JSON 确定性组装成最终 Prompt，不调用 LLM、不重新推理。
* `outfit_item.txt`：单个服装 / 鞋 / 包 / 配件的 Prompt 模板。
* `template.txt`：最终换装 Prompt 的全局模板。
* `__init__.py`：Feature 的导出和注册入口。

核心边界：

```text
Vision LLM = 理解图片
Schema = 结构化数据协议
Compiler = 确定性组装
Templates = Prompt 表达结构
Image Model = 执行最终换装
```

以下图片角色与优先级适用于 Replacement 预设。

Image 1 是固定主体锚点，同时可以作为兼容的 styling prior：它提供人物身份、姿势、构图、视角、四肢位置、接触点和自然遮挡，也可以在不与目标服装冲突时提示叠穿关系、塞入状态、袖口处理、腰部关系和配件摆放。Image 1 的原服装不是目标服装身份参考。
Images 2..N 定义目标 outfit 的身份、类别、版型、结构、颜色、图案、材质和具体细节，包括服装、鞋、包、腰带等。

优先级必须保持：

```text
Image 1 主体几何与构图
→ Images 2..N 的目标服装身份与结构
→ Image 1 的兼容 styling prior
→ 立体、自然、物理一致的服装表现
```

不要为了展示隐藏的衣物、下摆、鞋、配件或局部细节而改变 Image 1 的姿势、四肢位置、接触点、遮挡、服装覆盖、裁切或构图。Image 1 的背景、灯光和摄影环境默认不继承。

Reference 应遵循“最小充分引用”：整体版型使用必要的 `main_refs`，具体局部细节在 `key_details[].source_ref` 中绑定一个最清晰、最权威的来源；compiler 会从这些 source_ref 确定性派生补充细节引用，不再维护独立的 `detail_refs`。避免无意义的多图交叉引用。

V2 协议删除 `id`、`must_preserve` 和 `priority`。`shape` 表达固有版型结构，`key_details` 按重要性顺序表达超出 shape 的显著细节，`subject.styling` 仅承载 Image 1 的兼容穿搭先验。


资源由 preset manifest、system/user prompt、单一 schema.json 和可选模板组成。
Prompt Context 输出 system prompt、user prompt、JSON Schema 和 Prompt Context；
Schema Placement 只决定 schema 是否写入 system prompt，不配置 LLM 的 response format。
Prompt Builder 接收 Prompt Context 与 LLM response，使用 jsonschema 校验后按 compiler
路由；Garment 是当前已有的一个 compiler，不修改通用 LLM 节点。

Enhancement 保留原图人物、姿势、构图、背景、灯光、色调影调及已有穿搭整体轮廓；
参考图只用于原图中已有服装和配饰的可靠细节纠正、材质精修与组件补全，不新增独立单品。
允许必要的局部形态、遮挡、接触阴影变化，不改变手势或为新增物品虚构支撑结构。
没有可靠对应物的参考条目应省略；不自动添加参考模特的陪衬配饰。
subject.styling 表达需保留的已有穿搭关系，shape 仅提供结构上下文。
不要求逐项错误诊断；复杂 Logo、印花优先引用图片，不猜测文字。
