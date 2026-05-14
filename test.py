"""
Video Highlight Cutter — 基于 OCR 关键词的自动高光片段提取工具
通过 OCR 识别视频帧中的关键词，自动裁剪并拼接高光片段。

依赖安装:
  pip install -r requirements.txt

还需安装:
  1. Tesseract OCR: https://github.com/UB-Mannheim/tesseract/releases
  2. FFmpeg: https://github.com/GyanD/codexffmpeg/releases

路径配置（自动检测，也可通过环境变量覆盖）:
  set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
  set FFMPEG_CMD=D:\ffmpeg\bin\ffmpeg.exe

用法 (网页模式，推荐新手):
  python test.py

用法 (命令行模式):
  python test.py input.mp4 -o output.mp4 -k "first blood/double kill/triple kill"
  python test.py input.mp4 -o output.mp4 -k "击杀/双杀/五杀" --context 3
  python test.py input.mp4 -o output.mp4 -k "victory/defeat" --sample-fps 1

实用命令:
  python test.py --check          # 一键检查环境配置
  python test.py --verbose        # 启用详细调试日志
"""

import argparse
import cv2
import os
import subprocess
import sys
import time
import uuid
import shutil
import threading
import numpy as np
from pathlib import Path

VERSION = "1.1.0"

# ======================== 路径配置 ========================
# 优先级: 环境变量 > PATH 自动查找 > 常见安装路径
# 可通过环境变量 TESSERACT_CMD / FFMPEG_CMD 覆盖
_TESSERACT_PATHS = [
    r"?"
]
_FFMPEG_PATHS = [
    r"?",
]


def _find_executable(env_var, name, common_paths):
    """自动查找可执行文件路径"""
    # 1. 环境变量
    env_path = os.environ.get(env_var)
    if env_path and os.path.isfile(env_path):
        return env_path
    # 2. PATH 查找
    which = shutil.which(name)
    if which:
        return which
    # 3. 常见安装路径
    for p in common_paths:
        if os.path.isfile(p):
            return p
    # 4. 兜底返回 name，让 check_dependencies 给出错误提示
    return name


TESSERACT_CMD = _find_executable("TESSERACT_CMD", "tesseract", _TESSERACT_PATHS)
FFMPEG_CMD = _find_executable("FFMPEG_CMD", "ffmpeg", _FFMPEG_PATHS)

# OCR 图像预处理参数（可调优识别率）
# 增大 contrast_boost 可提升低对比度文字识别，但可能引入噪点
OCR_CONFIG = {
    "contrast_boost": 1.2,      # 对比度增强系数 (1.0=原图, >1.0=增强)
    "sharpen_kernel": 3,        # 锐化核大小 (0=禁用, 3/5/7)
    "denoise_strength": 5,      # 去噪强度 (0-10, 0=禁用)
}
# =========================================================

# 全局状态
_log_sink = []          # web 模式下收集日志
_verbose = False        # 是否启用详细日志
_start_time = None      # 用于统计耗时

# 终端颜色支持（Windows 10+ 自动启用）
class Colors:
    RED = '\033[91m' if sys.platform != 'win32' or os.environ.get('ANSICON') else ''
    GREEN = '\033[92m' if sys.platform != 'win32' or os.environ.get('ANSICON') else ''
    YELLOW = '\033[93m' if sys.platform != 'win32' or os.environ.get('ANSICON') else ''
    BLUE = '\033[94m' if sys.platform != 'win32' or os.environ.get('ANSICON') else ''
    RESET = '\033[0m' if sys.platform != 'win32' or os.environ.get('ANSICON') else ''

def log(msg, level="INFO", color=None):
    """增强日志: 带时间戳、级别、可选颜色"""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time
    timestamp = f"[{elapsed:5.1f}s]"
    level_str = f"[{level:7s}]"

    # 应用颜色
    if color and sys.stdout.isatty():
        msg = f"{color}{msg}{Colors.RESET}"

    # 格式化输出
    try:
        print(f"{timestamp} {level_str} {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"{timestamp} {level_str} {msg.encode('utf-8', errors='replace').decode('ascii', errors='replace')}", flush=True)

    # 收集日志 (web 模式)
    if _log_sink is not None:
        _log_sink.append(f"{timestamp} [{level}] {msg}")

    # 详细日志
    if _verbose and level == "DEBUG":
        pass  # 已在上方打印

def log_error(msg):
    log(msg, level="ERROR", color=Colors.RED)

def log_success(msg):
    log(msg, level="SUCCESS", color=Colors.GREEN)

def log_warning(msg):
    log(msg, level="WARN", color=Colors.YELLOW)

def log_debug(msg):
    if _verbose:
        log(msg, level="DEBUG", color=Colors.BLUE)

# ======================== 依赖检查 ========================

def _check_file_accessible(path, description):
    """检查文件是否存在 + 可读取"""
    if not os.path.isfile(path):
        return False, f"文件不存在: {path}"
    if not os.access(path, os.R_OK):
        return False, f"无读取权限: {path}"
    return True, None


def check_dependencies(detailed=False):
    """
    修复版依赖检查：兼容不同 pytesseract 版本 + 宽松判断 ffmpeg
    """
    errors = []
    log_debug("开始依赖检查...")

    # 1. Python 基础库
    for pkg, install_cmd in [("cv2", "opencv-python"), ("pytesseract", "pytesseract"),
                             ("numpy", "numpy"), ("flask", "flask")]:
        try:
            __import__(pkg)
            log_debug(f"✓ Python 包: {pkg}")
        except ImportError:
            err = f"缺少 Python 包: {pkg} (运行: pip install {install_cmd})"
            errors.append(err)
            log_error(err)

    # 2. Tesseract OCR 引擎 (修复 AttributeError)
    try:
        import pytesseract

        # 🔧 修复: 兼容新旧版本 pytesseract 的 tesseract_cmd 设置
        if TESSERACT_CMD:
            ok, err = _check_file_accessible(TESSERACT_CMD, "Tesseract")
            if not ok:
                raise RuntimeError(err)

            # 尝试两种设置方式 (新旧版本兼容)
            if hasattr(pytesseract, 'pytesseract'):
                # 旧版本: pytesseract.pytesseract.tesseract_cmd
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            else:
                # 新版本: pytesseract.tesseract_cmd
                pytesseract.tesseract_cmd = TESSERACT_CMD
            log_debug(f"✓ Tesseract 路径已设置: {TESSERACT_CMD}")

        # 获取版本 (实际调用二进制)
        version = pytesseract.get_tesseract_version()
        version_str = str(version).strip() if hasattr(version, 'strip') else str(version)
        log_success(f"Tesseract 引擎: {version_str}")

    except ImportError:
        err = "pytesseract 库未安装 (运行: pip install pytesseract)"
        errors.append(err)
        log_error(err)
    except AttributeError as e:
        # 🔧 专门捕获 AttributeError 并给出精准建议
        err = f"Tesseract 配置异常: {e}"
        err += "\n   💡 解决方案:"
        err += f"\n   1. 确认路径存在: {TESSERACT_CMD}"
        err += f"\n   2. 或用以下命令验证路径: python -c \"import os; print(os.path.isfile(r'{TESSERACT_CMD}'))\""
        err += f"\n   3. 如果路径正确，尝试更新 pytesseract: pip install --upgrade pytesseract"
        errors.append(err)
        log_error(err)
        if detailed:
            import traceback
            log_debug(f"   完整堆栈:\n{traceback.format_exc()}")
    except Exception as e:
        err = f"Tesseract 调用失败: {type(e).__name__}: {e}"
        if "not installed" in str(e).lower() or "not in your path" in str(e).lower():
            err += "\n   💡 解决方案:\n"
            err += f"   1. 确认文件存在: {TESSERACT_CMD}\n"
            err += f"   2. 或用 Shift+右键复制真实路径替换脚本中的 TESSERACT_CMD\n"
            err += f"   3. 或安装 64位 (w64) 版本 (与 Python 位数一致)"
        elif "winerror 193" in str(e).lower():
            err += " (32/64位不匹配)"
            err += "\n   💡 解决方案: 下载 Tesseract w64 版本覆盖安装"
        errors.append(err)
        log_error(err)

    # 3. FFmpeg (修复: 宽松判断，只要输出版本就认为成功)
    try:
        # 如果看起来是完整路径，先检查文件存在
        if os.sep in FFMPEG_CMD and not os.path.isfile(FFMPEG_CMD):
            raise FileNotFoundError(f"文件不存在: {FFMPEG_CMD}")

        # 不依赖 returncode，只要输出包含 "ffmpeg version" 就认为成功
        result = subprocess.run(
            [FFMPEG_CMD, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0  # Windows 隐藏黑窗
        )

        # 关键修复: 检查输出内容而非返回码
        if result.stdout and "ffmpeg version" in result.stdout.lower():
            version_line = result.stdout.splitlines()[0]
            log_success(f"FFmpeg: {version_line}")
        else:
            # 尝试 stderr (某些构建输出到 stderr)
            if result.stderr and "ffmpeg version" in result.stderr.lower():
                version_line = result.stderr.splitlines()[0]
                log_success(f"FFmpeg: {version_line}")
            else:
                raise RuntimeError(
                    f"未检测到版本信息 (stdout: {len(result.stdout)} chars, stderr: {len(result.stderr)} chars)")

    except Exception as e:
        err = f"FFmpeg 不可用: {type(e).__name__}: {e}"
        if "文件不存在" in str(e):
            err += "\n   💡 解决方案:\n"
            err += f"   1. 确认路径: {FFMPEG_CMD}\n"
            err += f"   2. 或用 Shift+右键复制 ffmpeg.exe 真实路径替换脚本中的 FFMPEG_CMD"
        elif "timeout" in str(e).lower():
            err += "\n   💡 解决方案: 杀毒软件可能拦截，尝试临时关闭或添加信任"
        errors.append(err)
        log_error(err)
        if detailed:
            log_debug(f"   stdout 前 200 字符: {result.stdout[:200] if 'result' in locals() else 'N/A'}")
            log_debug(f"   stderr 前 200 字符: {result.stderr[:200] if 'result' in locals() else 'N/A'}")

    # 4. OpenCV 基础验证
    try:
        import cv2
        test_img = cv2.cvtColor(np.zeros((10, 10, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        log_debug("✓ OpenCV 基础功能正常")
    except Exception as e:
        err = f"OpenCV 初始化失败: {e}"
        errors.append(err)
        log_error(err)

    # 汇总
    if errors:
        log_error(f"\n❌ 依赖检查未通过 ({len(errors)} 项失败)")
        if not detailed:
            log_warning("💡 运行 `python test.py --check --verbose` 查看完整诊断")
        return False, errors

    log_success("✅ 所有依赖检查通过")
    return True, []
# ======================== OCR 增强预处理 ========================

def enhance_image_for_ocr(image, config=None):
    """
    图像预处理流水线，提升 OCR 识别率
    :param image: BGR 格式的 numpy 数组
    :param config: OCR_CONFIG 字典
    :return: 预处理后的灰度二值图像
    """
    if config is None:
        config = OCR_CONFIG

    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 对比度增强 (CLAHE)
    if config["contrast_boost"] > 1.0:
        clahe = cv2.createCLAHE(clipLimit=config["contrast_boost"] * 2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 2. 锐化 (可选)
    if config["sharpen_kernel"] > 0:
        k = config["sharpen_kernel"]
        kernel = np.array([[-1, -1, -1],
                          [-1,  k*k+1, -1],
                          [-1, -1, -1]], dtype=np.float32) / (k*k)
        gray = cv2.filter2D(gray, -1, kernel)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # 3. 去噪 (可选)
    if config["denoise_strength"] > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=config["denoise_strength"])

    # 4. OTSU 二值化 (核心步骤)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def ocr_frame_text(frame, pytesseract, lang="chi_sim+eng"):
    """对帧进行 OCR 识别，返回小写文本 (使用增强预处理)"""
    binary = enhance_image_for_ocr(frame)
    # psm 6: 假设是单个均匀文本块，适合游戏内弹窗/提示
    text = pytesseract.image_to_string(binary, lang=lang, config="--psm 6")
    return text.lower().strip()

# ======================== 核心逻辑 (小幅优化) ========================

def detect_keywords_in_frame(frame, pytesseract, keywords_lower, roi_ratios, lang):
    """检测帧的 ROI 区域内是否包含关键词"""
    h, w = frame.shape[:2]
    t, b, l, r = roi_ratios

    y1 = int(h * t / 10)
    y2 = int(h * (10 - b) / 10)
    x1 = int(w * l / 10)
    x2 = int(w * (10 - r) / 10)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, []

    text = ocr_frame_text(roi, pytesseract, lang)
    if not text:
        return False, []

    matched = [kw for kw in keywords_lower if kw in text]
    return len(matched) > 0, matched

def merge_close_events(times, max_gap=3.0):
    """将时间上接近的事件点分组"""
    if not times:
        return []
    sorted_times = sorted(times)
    groups = [[sorted_times[0]]]
    for t in sorted_times[1:]:
        if t - groups[-1][-1] <= max_gap:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups

def merge_intervals(intervals, min_gap=0.5):
    """合并重叠或间距小于 min_gap 的时间区间"""
    if not intervals:
        return []
    sorted_i = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_i[0])]
    for s, e in sorted_i[1:]:
        if s <= merged[-1][1] + min_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def concat_with_ffmpeg(input_path, segments, output_path):
    """使用 ffmpeg filter_complex 拼接多个片段为单个视频 (添加错误详情)"""
    n = len(segments)
    if n == 0:
        raise ValueError("segments 不能为空")

    filter_parts = []
    v_tags, a_tags = [], []

    for i, (start, end) in enumerate(segments):
        dur = end - start
        if dur <= 0:
            log_warning(f"⚠️  跳过无效片段 [{i}]: {start:.2f}s - {end:.2f}s")
            continue
        filter_parts.append(f"[0:v]trim=start={start}:duration={dur},setpts=PTS-STARTPTS[v{i}];")
        filter_parts.append(f"[0:a]atrim=start={start}:duration={dur},asetpts=PTS-STARTPTS[a{i}];")
        v_tags.append(f"[v{i}]")
        a_tags.append(f"[a{i}]")

    if not v_tags:
        raise ValueError("所有片段时长无效，无法拼接")

    v_concat = "".join(v_tags) + f"concat=n={len(v_tags)}:v=1:a=0[vout];"
    a_concat = "".join(a_tags) + f"concat=n={len(a_tags)}:v=0:a=1[aout]"
    filter_complex = "".join(filter_parts) + v_concat + a_concat

    cmd = [
        FFMPEG_CMD,
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path,
    ]

    log("🎬 正在使用 ffmpeg 拼接高光片段...")
    log_debug(f"ffmpeg 命令: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        # 提取关键错误行
        stderr_lines = result.stderr.strip().split('\n')
        key_errors = [l for l in stderr_lines if 'error' in l.lower() or 'failed' in l.lower()]
        error_summary = '\n'.join(key_errors[-5:]) if key_errors else result.stderr[-500:]
        raise RuntimeError(f"ffmpeg 失败 (耗时 {elapsed:.1f}s):\n{error_summary}")

    log_success(f"拼接完成 (耗时 {elapsed:.1f}s) | 输出: {os.path.abspath(output_path)}")

def extract_highlights(
    input_path,
    output_path,
    keywords,
    context_sec=4.0,
    merge_gap_sec=3.0,
    roi_ratios=(1, 1, 1, 1),
    sample_fps=2.0,
    lang="chi_sim+eng",
):
    """
    主流程：扫描视频 → OCR 检测关键词 → 裁剪拼接高光片段
    """
    import pytesseract
    import numpy as np  # OCR 增强需要

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    keywords_lower = [k.lower() for k in keywords]
    log(f"🔧 关键词: {keywords} | 语言: {lang}")

    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log_error(f"无法打开视频: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        log_error("视频元数据无效 (FPS 或帧数异常)")
        return False

    duration = total_frames / fps
    frame_interval = max(1, int(round(fps / sample_fps)))
    total_to_process = max(1, total_frames // frame_interval)

    log(f"📊 视频: {fps:.1f}FPS | {total_frames}帧 | {duration:.1f}秒")
    log(f"📷 OCR 采样: {sample_fps}FPS (每 {frame_interval} 帧检测一次，共 ~{total_to_process} 次)")

    # === 阶段 1：扫描 ===
    trigger_times = []
    frame_idx = 0
    processed = 0
    last_log_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        current_sec = frame_idx / fps

        try:
            is_match, matched = detect_keywords_in_frame(
                frame, pytesseract, keywords_lower, roi_ratios, lang
            )
        except Exception as e:
            log_debug(f"OCR 异常 (帧 {frame_idx}): {e}")
            frame_idx += 1
            continue

        if is_match:
            log(f"  [{current_sec:.1f}s] ✅ {', '.join(matched)}")
            trigger_times.append(current_sec)

        frame_idx += 1
        processed += 1

        # 进度日志 (每 10% 或 5 秒输出一次)
        now = time.time()
        if processed % max(1, total_to_process // 10) == 0 or (now - last_log_time) > 5:
            pct = min(100, processed / total_to_process * 100)
            elapsed = now - last_log_time
            last_log_time = now
            log(f"⏳ 进度: {pct:.0f}% ({processed}/{total_to_process})")

    cap.release()

    if not trigger_times:
        log_warning("❌ 未检测到任何高光片段")
        log_warning("💡 排查建议:")
        log_warning("   1. 关键词是否出现在视频中？尝试 --sample-fps 1.0 提高采样")
        log_warning("   2. 文字位置是否在 ROI 内？尝试 --roi 0,0,0,0 检测全屏")
        log_warning("   3. OCR 是否识别成功？用 --verbose 查看原始识别文本")
        return False

    # === 阶段 2：合并事件生成时间区间 ===
    groups = merge_close_events(trigger_times, max_gap=merge_gap_sec)
    segments = []
    for g in groups:
        s = max(0.0, min(g) - context_sec)
        e = min(duration, max(g) + context_sec)
        segments.append((s, e))
    segments = merge_intervals(segments, min_gap=0.5)

    log(f"\n🎯 {len(trigger_times)} 个命中点 → 合并为 {len(segments)} 个片段")
    for i, (s, e) in enumerate(segments):
        log(f"  [{i+1}] {s:.1f}s - {e:.1f}s (时长 {e-s:.1f}s)")

    # === 阶段 3：拼接 ===
    concat_with_ffmpeg(input_path, segments, output_path)
    return True

# ======================== Web 模式 (小幅优化) ========================

TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)

def wait_for_file_ready(file_path, timeout=30.0, check_interval=0.5):
    """增强版文件就绪检查 (更长超时 + 更短间隔)"""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:  # 至少 1KB
            try:
                with open(file_path, "rb") as f:
                    f.read(1)
                return True
            except (OSError, PermissionError):
                pass
        time.sleep(check_interval)
    return False

def run_web_server(host="127.0.0.1", port=5000):
    """启动 Flask 网页服务"""
    try:
        from flask import Flask, request, jsonify, send_file, render_template
    except ImportError:
        log_error("缺少 flask，请运行: pip install flask")
        sys.exit(1)

    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

    DEFAULT_FORM = {
        "top": 1, "bottom": 1, "left": 1, "right": 1,
        "context_sec": 4.0, "merge_gap_sec": 3.0, "sample_fps": 2.0,
        "keywords_text": "", "lang": "chi_sim+eng",
    }

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", form_data=DEFAULT_FORM, error=None, message=None)

    @app.route("/process", methods=["POST"])
    def process():
        global _log_sink
        _log_sink = []
        _start_time = time.time()  # 重置计时

        try:
            keywords_text = request.form.get("keywords_text", "").strip()
            context_sec   = float(request.form.get("context_sec", 4.0))
            merge_gap_sec = float(request.form.get("merge_gap_sec", 3.0))
            sample_fps    = float(request.form.get("sample_fps", 2.0))
            lang          = request.form.get("lang", "chi_sim+eng").strip() or "chi_sim+eng"
            top    = int(request.form.get("top", 1))
            bottom = int(request.form.get("bottom", 1))
            left   = int(request.form.get("left", 1))
            right  = int(request.form.get("right", 1))

            keywords = [k.strip() for k in keywords_text.split("/") if k.strip()]
            if not keywords:
                return jsonify({"success": False, "error": "请至少输入一个关键词（用 / 分隔）", "logs": []})

            video_file = request.files.get("video")
            if not video_file or not video_file.filename:
                return jsonify({"success": False, "error": "未上传视频文件", "logs": []})

            input_path  = TEMP_DIR / f"input_{uuid.uuid4().hex}.mp4"
            output_path = TEMP_DIR / f"output_{uuid.uuid4().hex}.mp4"
            video_file.save(str(input_path))
            log_debug(f"Web 上传: {input_path} → {output_path}")

            # 依赖检查
            ok, errs = check_dependencies()
            if not ok:
                return jsonify({
                    "success": False,
                    "error": "依赖检查失败，请查看控制台输出",
                    "logs": _log_sink,
                })

            roi = (
                max(0, min(10, top)),
                max(0, min(10, bottom)),
                max(0, min(10, left)),
                max(0, min(10, right)),
            )

            ok = extract_highlights(
                input_path=str(input_path),
                output_path=str(output_path),
                keywords=keywords,
                context_sec=context_sec,
                merge_gap_sec=merge_gap_sec,
                roi_ratios=roi,
                sample_fps=sample_fps,
                lang=lang,
            )

            if not ok or not wait_for_file_ready(str(output_path)):
                return jsonify({
                    "success": False,
                    "error": "未检测到高光片段或输出文件生成失败",
                    "logs": _log_sink,
                })

            stem = output_path.stem
            return jsonify({
                "success": True,
                "download_url": f"/download/{stem}",
                "logs": _log_sink,
            })
        except Exception as e:
            log_error(f"Web 处理异常: {e}")
            return jsonify({"success": False, "error": str(e), "logs": _log_sink})
        finally:
            # 清理临时输入文件
            if 'input_path' in locals() and Path(input_path).exists():
                try:
                    Path(input_path).unlink()
                except Exception as e:
                    log_debug(f"清理输入文件失败: {e}")

    @app.route("/download/<stem>")
    def download(stem):
        file_path = TEMP_DIR / f"{stem}.mp4"
        if not file_path.exists():
            return "文件不存在或已过期", 404
        copy_path = TEMP_DIR / f"{stem}_dl_{uuid.uuid4().hex[:6]}.mp4"
        shutil.copy2(str(file_path), str(copy_path))

        def _cleanup():
            time.sleep(600)
            for p in [file_path, copy_path]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

        threading.Thread(target=_cleanup, daemon=True).start()
        return send_file(str(copy_path), as_attachment=True, download_name="highlight.mp4", mimetype="video/mp4")

    url = f"http://{host}:{port}"
    log_success(f"🌐 高光生成器已启动: {url}")
    log("   按 Ctrl+C 停止服务")

    import webbrowser
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, threaded=True)

# ======================== CLI / 入口 ========================

def run_env_check():
    """--check 模式: 仅检查环境并退出"""
    log(f"🔍 环境诊断模式 v{VERSION}")
    log(f"   Python: {sys.version.split()[0]} | 平台: {sys.platform}")
    log(f"   脚本路径: {os.path.abspath(__file__)}")
    log(f"   TESSERACT_CMD: {TESSERACT_CMD}")
    log(f"   FFMPEG_CMD: {FFMPEG_CMD}\n")

    ok, errors = check_dependencies(detailed=_verbose)

    if ok:
        log_success("\n✅ 环境配置正确，可正常运行!")
        log("💡 下一步:")
        log("   • 网页模式: 直接运行 `python test.py`")
        log("   • 命令行: `python test.py input.mp4 -k \"关键词\" -o out.mp4`")
        return 0
    else:
        log_error(f"\n❌ 环境检查失败 ({len(errors)} 项)")
        return 1

def main():
    global _verbose

    # 解析参数 (支持 --check / --verbose)
    parser = argparse.ArgumentParser(
        description="Video Highlight Cutter — 基于 OCR 关键词的高光片段自动提取 (优化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 网页模式 (推荐新手)
  python test.py

  # 命令行模式 - 英文关键词
  python test.py game.mp4 -k "first blood/double kill" -o highlights.mp4

  # 命令行模式 - 中文关键词 + 调整参数
  python test.py game.mp4 -k "击杀/五杀" --context 5 --sample-fps 1 --roi 0,2,0,0

  # 仅检查环境
  python test.py --check

  # 检查环境 + 详细日志
  python test.py --check --verbose
        """,
    )
    parser.add_argument("input", nargs="?", help="输入视频文件 (网页模式可省略)")
    parser.add_argument("-o", "--output", default="highlight_output.mp4", help="输出视频 (默认: highlight_output.mp4)")
    parser.add_argument("-k", "--keywords", help="关键词, 用 / 分隔, 如: first blood/double kill/击杀")
    parser.add_argument("--context", type=float, default=4.0, help="命中点前后保留秒数 (默认: 4.0)")
    parser.add_argument("--merge-gap", type=float, default=3.0, help="事件合并最大间隔秒数 (默认: 3.0)")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="OCR 采样帧率 (默认: 2.0)")
    parser.add_argument("--roi", type=str, default="1,1,1,1", help="ROI top,bottom,left,right (0-10, 默认: 1,1,1,1)")
    parser.add_argument("--lang", default="chi_sim+eng", help="Tesseract 语言包 (默认: chi_sim+eng)")

    # 版本信息
    parser.add_argument("--version", action="store_true", help="显示版本号")

    # 调试参数
    parser.add_argument("--check", action="store_true", help="仅检查环境配置，不处理视频")
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细调试日志")

    args = parser.parse_args()
    _verbose = args.verbose

    # --version 模式
    if args.version:
        print(f"Video Highlight Cutter v{VERSION}")
        sys.exit(0)

    # --check 模式
    if args.check:
        sys.exit(run_env_check())

    # 无参数 → 网页模式
    if args.input is None:
        # 网页模式也先快速检查关键依赖
        ok, _ = check_dependencies()
        if not ok:
            log_error("依赖检查失败，请运行 `python test.py --check` 查看详情")
            sys.exit(1)
        run_web_server()
        return

    # 命令行模式参数验证
    if not args.keywords:
        log_error("命令行模式必须指定 -k/--keywords")
        sys.exit(1)

    if not check_dependencies()[0]:
        sys.exit(1)

    if not os.path.isfile(args.input):
        log_error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    keywords = [k.strip() for k in args.keywords.split("/") if k.strip()]
    if not keywords:
        log_error("未提供有效关键词")
        sys.exit(1)

    try:
        roi = tuple(int(x) for x in args.roi.split(","))
        assert len(roi) == 4 and all(0 <= x <= 10 for x in roi)
    except (ValueError, AssertionError):
        log_error("ROI 格式无效，请使用 top,bottom,left,right 各 0-10 (例: --roi 0,2,0,0)")
        sys.exit(1)

    log(f"🚀 开始处理: {args.input}")
    start = time.time()

    ok = extract_highlights(
        input_path=args.input,
        output_path=args.output,
        keywords=keywords,
        context_sec=args.context,
        merge_gap_sec=args.merge_gap,
        roi_ratios=roi,
        sample_fps=args.sample_fps,
        lang=args.lang,
    )

    elapsed = time.time() - start
    if ok:
        log_success(f"\n🎉 处理完成! (总耗时 {elapsed:.1f}s)")
        log_success(f"📦 输出文件: {os.path.abspath(args.output)}")
    else:
        log_error(f"\n❌ 处理失败 (耗时 {elapsed:.1f}s)")

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()