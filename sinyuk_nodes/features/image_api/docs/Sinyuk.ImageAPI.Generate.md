# Image Generate

使用 1-10 张参考图执行一次图片编辑任务。

- 图片按输入顺序共同发送，不会逐张生成。
- 可选连接 Image Ratio 的比例输出覆盖模型手动比例；未连接时使用手动值。外部比例当前仅支持 RunningHub，并由模型请求适配器按支持比例进行等价匹配。
- 每张 PNG 不超过 10 MB，总计不超过 50 MB。
- 输出可连接 Preview Image、Save Image 或其他图片节点。
