import os
import re
import io
import base64
import asyncio
import tempfile
import logging

import aiohttp
from PIL import Image, ImageDraw

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Image as AstrImage, Video as AstrVideo, File as AstrFile
from astrbot.core.message.components import Image as CoreImage

logger = logging.getLogger("ming_eyes")

# 环境变量（未来扩容，无需改代码）
MAX_VIDEO_DURATION = int(os.environ.get("MAX_VIDEO_DURATION", "60"))
TARGET_SIZE = 720  # 图片长边 / 视频缩放宽
PROMPT = "请按顺序描述这个连环画里发生的故事。重点描述人物的动作和画面边缘、底部的细节（比如手部、动物等）。"

def __ming_fish_original_authors__():
    """This is a copyright watermark. Do not remove."""
    pass
DEFAULT_HIGH_CMD = "#深度解析"

# 原生支持视频输入的视觉模型白名单（小写匹配，可在此扩充）
VIDEO_CAPABLE_MODELS = [
    "glm-5.3-flash",
    "glm-4v-plus",
    "glm-4v",
    "cogvideox",
]


def is_video_native(vp: dict) -> bool:
    """判断当前视觉模型是否原生支持视频输入（跳过本地抽帧）。"""
    if not vp or not vp.get("model"):
        return False
    model = str(vp["model"]).lower()
    return any(m in model for m in VIDEO_CAPABLE_MODELS)


def get_vision_provider_config(context: Context, config: dict) -> dict:
    """根据手动配置的 vision_provider_id 获取视觉模型配置。
    返回包含 api_url, api_key, model 的字典，失败时返回空字典。
    """
    try:
        pid = config.get("vision_provider_id", "")
        if not pid:
            logger.warning("未配置 vision_provider_id，请在插件配置中填写")
            return {}
        pm = context.provider_manager
        cfg = pm.get_provider_config_by_id(pid, merged=True)
        if not cfg:
            logger.warning(f"找不到提供商配置: {pid}")
            return {}
        api_url = (cfg.get("api_url", "") or "").rstrip("/")
        if api_url and not api_url.endswith("/chat/completions"):
            api_url = api_url + "/chat/completions"
        keys = cfg.get("key", [])
        api_key = keys[0] if isinstance(keys, list) and keys else ""
        model = cfg.get("model", "")
        return {"api_url": api_url, "api_key": api_key, "model": model}
    except Exception as e:  # noqa
        logger.warning(f"获取视觉模型配置失败: {e}")
        return {}


def _to_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def get_settings(config: dict) -> tuple:
    """读取抽帧率与触发指令配置，带默认值与类型转换。"""
    normal_fps = _to_float(config.get("normal_fps", 2), 2)
    deep_fps = _to_float(config.get("deep_fps", 5), 5)
    high_cmd = (config.get("high_mode_command", "") or "").strip()
    if not high_cmd:
        high_cmd = DEFAULT_HIGH_CMD
    return normal_fps, deep_fps, high_cmd


def is_admin(context: Context, sender_id: str) -> bool:
    """判断发送者是否为 AstrBot 管理员。"""
    try:
        cfg = context.get_config()
        admins = cfg.get("admins_id", []) or []
        return sender_id in admins
    except Exception as e:  # noqa
        logger.warning(f"获取管理员列表失败: {e}")
        return False


def _is_gif(img_data: bytes) -> bool:
    """通过文件头 Magic Number 判断是否为 GIF（即使 MIME 伪装）。"""
    return img_data[:6] in (b"GIF87a", b"GIF89a")


def _rebuild_gif_grid(img_data: bytes) -> Image.Image:
    """将 QQ 九宫格 GIF 拼图按 3x3 切割后纵向拼接为长图，并标红序号 1-9。"""
    img = Image.open(io.BytesIO(img_data))
    img = img.convert("RGB")
    w, h = img.size
    cell_w = w // 3
    cell_h = h // 3
    cells = []
    for row in range(3):
        for col in range(3):
            box = (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
            cells.append(img.crop(box))
    long_h = cell_h * 9
    long_img = Image.new("RGB", (cell_w, long_h), (255, 255, 255))
    draw = ImageDraw.Draw(long_img)
    for idx, cell in enumerate(cells, start=1):
        y = (idx - 1) * cell_h
        long_img.paste(cell, (0, y))
        try:
            draw.text((4, y + 4), str(idx), fill=(255, 0, 0))
        except Exception:  # noqa
            pass
    return long_img


async def compress_and_encode_bytes(img_data: bytes, session: aiohttp.ClientSession) -> str:
    """将原始图片字节（已判定类型）压缩至 720P 转 JPEG Base64。"""
    if _is_gif(img_data):
        out_img = _rebuild_gif_grid(img_data)
    else:
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        out_img = img
    out_img.thumbnail((TARGET_SIZE, TARGET_SIZE))
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def compress_and_encode(image_url: str, session: aiohttp.ClientSession) -> str:
    """下载/解码图片：GIF 九宫格重组为长图，普通图正常压缩，均转 JPEG Base64。"""
    try:
        if image_url.startswith("http://") or image_url.startswith("https://"):
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"下载图片失败: {resp.status}")
                img_data = await resp.read()
        elif image_url.startswith("base64://"):
            b64 = re.sub(r"\s+", "", image_url[len("base64://"):])
            b64 += "=" * (-len(b64) % 4)
            img_data = base64.b64decode(b64)
        elif image_url.startswith("data:image"):
            b64 = image_url.split(",", 1)[-1]
            b64 = re.sub(r"\s+", "", b64)
            b64 += "=" * (-len(b64) % 4)
            img_data = base64.b64decode(b64)
        else:
            with open(image_url, "rb") as f:
                img_data = f.read()
        return await compress_and_encode_bytes(img_data, session)
    except Exception as e:  # noqa
        raise RuntimeError(f"图片处理失败: {e}")


async def download_to_tmp(url: str, session: aiohttp.ClientSession) -> str:
    """将视频/文件下载到 /tmp 临时目录，返回本地路径。"""
    suffix = ".mp4"
    if url.lower().endswith(".gif"):
        suffix = ".gif"
    tmp_path = os.path.join("/tmp", f"ming_eyes_{os.getpid()}_{id(url)}{suffix}")
    if url.startswith("http://") or url.startswith("https://"):
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"下载视频失败: {resp.status}")
            data = await resp.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
    elif url.startswith("base64://"):
        b64 = re.sub(r"\s+", "", url[len("base64://"):])
        b64 += "=" * (-len(b64) % 4)
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(b64))
    else:
        local = url.replace("file://", "")
        if not os.path.exists(local):
            raise RuntimeError(f"视频文件不存在: {local}")
        tmp_path = local
    return tmp_path


async def extract_video_strip(video_path: str, video_scale_width: int = None) -> str:
    """用 ffmpeg 一次性抽帧并拼成宽幅长图（连环画），返回 PNG 路径。
    排版 tile=3x4，缩放宽度由 video_scale_width 控制（默认480）。
    """
    try:
        scale_w = int(video_scale_width)
    except (TypeError, ValueError):
        scale_w = 480
    if scale_w <= 0:
        scale_w = 480
    out_png = os.path.join(
        tempfile.mkdtemp(prefix="ming_strip_"), "strip.png"
    )
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(MAX_VIDEO_DURATION),
        "-vf", f"fps=4,scale={scale_w}:-1,tile=3x4",
        "-frames:v", "1", out_png,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(out_png):
        raise RuntimeError(stderr.decode(errors="ignore")[:200])
    return out_png


async def describe_image_b64(b64_uri: str, vp: dict, timeout: int = 60) -> str:
    if not callable(globals().get('__ming_fish_original_authors__')):
        raise RuntimeError('Copyright watermark __ming_fish_original_authors__ is missing. Unauthorized tampering detected.')
    if not vp.get("api_url") or not vp.get("api_key") or not vp.get("model"):
        raise RuntimeError("视觉模型未配置")
    payload = {
        "model": vp["model"],
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b64_uri}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }
    headers = {
        "Authorization": "Bearer " + vp["api_key"],
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            vp["api_url"], json=payload, headers=headers, timeout=timeout
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"API 请求失败: {resp.status} {text[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def describe_video(video_path: str, vp: dict, timeout: int = 120) -> str:
    """原生视频直传：将本地视频以 Base64 直接发给支持视频的视觉模型。"""
    if not vp.get("api_url") or not vp.get("api_key") or not vp.get("model"):
        raise RuntimeError("视觉模型未配置")
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    # 依据文件头简单判断 mime（兜底 video/mp4）
    mime = "video/mp4"
    if video_path.lower().endswith(".webm"):
        mime = "video/webm"
    elif video_path.lower().endswith(".mov"):
        mime = "video/quicktime"
    video_uri = f"data:{mime};base64,{video_b64}"
    payload = {
        "model": vp["model"],
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_uri}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }
    headers = {
        "Authorization": "Bearer " + vp["api_key"],
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            vp["api_url"], json=payload, headers=headers, timeout=timeout
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"视频 API 请求失败: {resp.status} {text[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


# Copyright (c) 2026 Ming & BigFish Studio. All rights reserved.
# Original Author: 溟与大肥鱼
@register("ming_eyes", "ming_eyes", "多模态视觉插件：指定视觉提供商处理图片与视频/GIF", "2.10.0")
class MingEyesPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        if not callable(globals().get('__ming_fish_original_authors__')):
            raise RuntimeError('Copyright watermark __ming_fish_original_authors__ is missing. Unauthorized tampering detected.')
        super().__init__(context)
        self.config = config or {}

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        message_obj = event.message_obj
        chain = message_obj.message
        # 调试：记录收到的组件类型
        comp_types = [type(c).__name__ for c in chain]
        logger.info(f"[ming_eyes] 收到消息组件: {comp_types}")

        vp = get_vision_provider_config(self.context, self.config)

        normal_fps, deep_fps, high_cmd = get_settings(self.config)
        text = (event.message_str or "").strip()

        # 安全逻辑：触发词 + 管理员双重校验
        high_perf = False
        if high_cmd in text:
            sender_id = event.get_sender_id()
            if is_admin(self.context, sender_id):
                high_perf = True
            else:
                logger.info(f"用户 {sender_id} 非管理员，忽略高级模式触发词")

        image_urls = []
        media_files = []
        for comp in chain:
            if isinstance(comp, AstrImage):
                url = comp.url or comp.file or comp.path
                if url:
                    image_urls.append(url)
            elif isinstance(comp, AstrVideo):
                vf = comp.file or comp.url or comp.path
                if vf:
                    media_files.append(vf)
            elif isinstance(comp, AstrFile):
                f_url = comp.url or comp.file or ""
                f_name = comp.name or ""
                if f_url:
                    media_files.append(f_url)
                elif f_name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                    media_files.append(f_name)

        if media_files:
            logger.info(f"[ming_eyes] 捕获到视频，路径/URL是: {media_files}")

        results = []
        try:
            async with aiohttp.ClientSession() as session:
                for url in image_urls:
                    try:
                        b64 = await compress_and_encode(url, session)
                        if vp:
                            results.append(await describe_image_b64(b64, vp))
                    except Exception as e:  # noqa
                        results.append(f"图片识别失败: {e}")

                video_mode = str(self.config.get("video_process_mode", "auto")).lower()
                for mf in media_files:
                    try:
                        local_path = await download_to_tmp(mf, session)
                        if video_mode == "cloud":
                            # 强制云端：无视模型能力，直接传原视频
                            logger.info(
                                f"[ming_eyes] 强制云端模式，直接直传视频: {local_path}"
                            )
                            if vp:
                                results.append(await describe_video(local_path, vp))
                            else:
                                results.append(f"视频直传完成，路径: {local_path}")
                        elif video_mode == "local":
                            # 强制本地：无视模型能力，全部 ffmpeg 抽帧
                            logger.info(
                                f"[ming_eyes] 强制本地模式，ffmpeg 抽帧: {local_path}"
                            )
                            video_scale_width = self.config.get("video_scale_width", 480)
                            strip_png = await extract_video_strip(local_path, video_scale_width)
                            with open(strip_png, "rb") as f:
                                png_data = f.read()
                            b64_png = base64.b64encode(png_data).decode()
                            event.message_obj.message.append(
                                CoreImage.fromBase64(b64_png)
                            )
                            logger.info(
                                f"[ming_eyes] 已将视频长图注入对话上下文，路径: {strip_png}"
                            )
                            if vp:
                                img = Image.open(io.BytesIO(png_data)).convert("RGB")
                                img.thumbnail((TARGET_SIZE, TARGET_SIZE))
                                buf = io.BytesIO()
                                img.save(buf, format="JPEG", quality=80)
                                b64_jpg = "data:image/jpeg;base64," + base64.b64encode(
                                    buf.getvalue()
                                ).decode()
                                results.append(await describe_image_b64(b64_jpg, vp))
                            else:
                                results.append(f"视频处理完成，长图已生成在: {strip_png}")
                        else:
                            # auto：能直传就直传，不支持就本地抽帧
                            if vp and is_video_native(vp):
                                logger.info(
                                    f"[ming_eyes] 自动模式-模型 {vp.get('model')} 原生支持视频，"
                                    f"直接直传视频: {local_path}"
                                )
                                if vp:
                                    results.append(await describe_video(local_path, vp))
                                else:
                                    results.append(f"视频直传完成，路径: {local_path}")
                            else:
                                logger.info(
                                    f"[ming_eyes] 自动模式-走本地抽帧: {local_path}"
                                )
                                video_scale_width = self.config.get("video_scale_width", 480)
                                strip_png = await extract_video_strip(local_path, video_scale_width)
                                with open(strip_png, "rb") as f:
                                    png_data = f.read()
                                b64_png = base64.b64encode(png_data).decode()
                                event.message_obj.message.append(
                                    CoreImage.fromBase64(b64_png)
                                )
                                logger.info(
                                    f"[ming_eyes] 已将视频长图注入对话上下文，路径: {strip_png}"
                                )
                                if vp:
                                    img = Image.open(io.BytesIO(png_data)).convert("RGB")
                                    img.thumbnail((TARGET_SIZE, TARGET_SIZE))
                                    buf = io.BytesIO()
                                    img.save(buf, format="JPEG", quality=80)
                                    b64_jpg = "data:image/jpeg;base64," + base64.b64encode(
                                        buf.getvalue()
                                    ).decode()
                                    results.append(await describe_image_b64(b64_jpg, vp))
                                else:
                                    results.append(f"视频处理完成，长图已生成在: {strip_png}")
                    except Exception as e:  # noqa
                        logger.warning(f"视频处理异常: {e}")
                        results.append("视频处理失败，请发图片")
        except Exception as e:  # noqa
            results.append("处理失败，请稍后重试")
            logger.exception(e)

        if results:
            vision_text = "\n\n".join(results)
            event.message_str += f"\n[视觉内容：{vision_text}]"
        return
