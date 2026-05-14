# Video Highlight Cutter

基于 OCR 关键词检测的**视频高光片段自动提取工具**。上传视频，输入关键词，自动裁剪出包含关键词的所有高光片段并拼接成一个视频。

适用场景：游戏精彩集锦、课程重点剪辑、会议记录提取等。
（持续优化中）
## 功能特性

- **双模式运行**：网页模式（拖拽上传） + 命令行模式（批量处理）
- **智能 OCR 检测**：支持中英文关键词混合识别
- **ROI 区域裁剪**：只扫描画面指定区域，提升识别精度与速度
- **自适应上下文**：命中关键词前后保留可配置的缓冲时长
- **智能合并**：时间上接近的高光片段自动合并，避免大量短片段
- **图像预处理增强**：CLAHE 对比度增强 + 锐化 + 去噪 + OTSU 二值化，提升 OCR 识别率
- **全面的依赖检查**：`python test.py --check` 一键诊断环境

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

或手动安装（要求 Python 3.8+）：

```bash
pip install opencv-python==4.10.0.84 pytesseract==0.3.13 numpy==1.26.4 flask==3.0.3
```

### 2. 安装 Tesseract OCR 引擎

下载并安装 [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/releases)（选择 64 位版本 w64）。

> 安装时务必勾选需要的语言包（简体中文 `chi_sim`、英文 `eng` 等）。

安装后，将 Tesseract 路径配置到环境变量 `TESSERACT_CMD`，或在 `test.py` 中修改 `TESSERACT_CMD` 变量：

```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # 改为你的实际路径
```

### 3. 安装 FFmpeg

下载 [FFmpeg](https://github.com/GyanD/codexffmpeg/releases) 并解压，将 `bin` 目录加入系统 PATH。

或是将路径配置到环境变量 `FFMPEG_CMD`，或在 `test.py` 中修改 `FFMPEG_CMD` 变量：

```python
FFMPEG_CMD = r"D:\ffmpeg\bin\ffmpeg.exe"  # 改为你的实际路径
```

验证安装：

```bash
python test.py --check
```

## 使用方法

### 网页模式（推荐新手）

```bash
python test.py
```

自动打开浏览器访问 `http://127.0.0.1:5000`，上传视频、输入关键词、点击生成即可。

### 命令行模式

```bash
# 英文关键词
python test.py game.mp4 -k "first blood/double kill/triple kill" -o highlights.mp4

# 中文关键词
python test.py game.mp4 -k "击杀/双杀/五杀" --context 3

# 胜利/失败检测
python test.py game.mp4 -k "victory/defeat" --sample-fps 1

# 自定义扫描区域（全屏检测）
python test.py game.mp4 -k "legendary" --roi 0,0,0,0
```

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频文件路径 | — |
| `-o, --output` | 输出视频路径 | `highlight_output.mp4` |
| `-k, --keywords` | 关键词，用 `/` 分隔 | — |
| `--context` | 命中点前后保留秒数 | `4.0` |
| `--merge-gap` | 事件合并最大间隔（秒） | `3.0` |
| `--sample-fps` | OCR 采样帧率 | `2.0` |
| `--roi` | 扫描区域 top,bottom,left,right (0-10) | `1,1,1,1` |
| `--lang` | Tesseract 语言包 | `chi_sim+eng` |
| `--check` | 仅检查环境配置 | — |
| `--verbose, -v` | 启用详细调试日志 | — |

## FAQ

**Q: 没有检测到任何高光？**
- 确认 Tesseract 安装正确并已安装对应语言包
- 试试 `--roi 0,0,0,0` 检测全屏
- 降低采样帧率 `--sample-fps 1` 提高每帧质量
- 用 `--verbose` 查看详细识别日志

**Q: OCR 识别率低？**
- 调节 `OCR_CONFIG` 中的对比度增强和锐化参数
- 确保画面中文字清晰、对比度充足
- 安装正确的 Tesseract 语言包（中英混合用 `chi_sim+eng`）

**Q: 处理速度慢？**
- 降低 `--sample-fps`（设为 1.0 可显著加速）
- 缩小 ROI 区域减少检测面积

**Q: 提示 32/64 位不匹配？**
- Python 和 Tesseract 必须同为 64 位，请下载 `w64` 版本的 Tesseract

## 项目结构

```
Video_Cut/
├── test.py              # 主程序入口
├── requirements.txt     # Python 依赖
├── templates/
│   └── index.html       # Web 页面
├── temp/                # 临时文件（自动生成）
├── _test_results/       # 测试结果（自动生成）
├── LICENSE
├── README.md
└── .gitignore
```

## 技术原理

1. **视频解析** — OpenCV 按指定帧率逐帧读取视频
2. **图像预处理** — CLAHE 增强 → 锐化 → 去噪 → OTSU 二值化
3. **OCR 检测** — Tesseract 识别指定 ROI 区域文本
4. **事件合并** — 时间轴命中点聚类 + 上下文扩展 + 区间合并
5. **视频拼接** — FFmpeg filter_complex 无缝拼接高光片段

## License

[MIT](LICENSE)
