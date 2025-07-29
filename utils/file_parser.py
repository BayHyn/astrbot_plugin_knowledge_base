from typing import Optional
from astrbot.api import logger
from astrbot.api.star import Context
import os
import base64
import aiofiles
from aiofiles.os import stat as aio_stat
import chardet
import asyncio
from ..core.constants import (
    COMMON_ENCODINGS,
    READ_FILE_LIMIT,
    TEXT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MARKITDOWN_EXTENSIONS,
    AUDIO_EXTENSIONS,
)
from markitdown_no_magika import MarkItDown
from openai import AsyncOpenAI, OpenAI


async def _detect_and_read_file(file_path: str) -> str:
    """
    检测文件编码并读取文件内容
    """
    content = None
    detected_encoding = None

    # 优化：对于非常大的文件，chardet 读取整个文件可能不理想
    # 可以先读取头部一小部分来检测
    try:
        file_size = (await aio_stat(file_path)).st_size
        read_limit = min(file_size, READ_FILE_LIMIT)

        async with aiofiles.open(file_path, "rb") as f_binary:
            raw_head = await f_binary.read(read_limit)  # 读取头部

        if raw_head:
            result = chardet.detect(raw_head)
            detected_encoding = result["encoding"]
            confidence = result["confidence"]

            if detected_encoding and confidence > 0.7:
                logger.info(
                    f"Chardet: {file_path} 编码={detected_encoding}, 置信度={confidence:.2f}"
                )
                try:
                    # 如果 chardet 成功，用检测到的编码完整读取文件
                    async with aiofiles.open(
                        file_path, "r", encoding=detected_encoding, errors="ignore"
                    ) as f:  # errors='ignore' 或 'replace' 可以增加容错
                        content = await f.read()
                    return content
                except UnicodeDecodeError:
                    logger.warning(
                        f"使用 Chardet 检测到的编码 {detected_encoding} 无法完整读取 {file_path}。尝试常用编码列表。"
                    )
                    content = None  # 确保回退
                except Exception as e_read_full:
                    logger.warning(
                        f"读取 {file_path} 时使用 Chardet 检测到的编码 {detected_encoding} 出错: {e_read_full}。尝试常用编码列表。"
                    )
                    content = None
            else:
                logger.info(
                    f"Chardet 对 {file_path} 的检测结果不确定 (编码: {detected_encoding}, 置信度: {confidence:.2f})。尝试常用编码列表。"
                )
        else:  # 文件为空或非常小
            logger.info(
                f"文件 {file_path} 为空或太小，无法进行 Chardet 检测。尝试常用编码列表。"
            )

    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise
    except Exception as e_chardet:
        logger.warning(
            f"对 {file_path} 进行 Chardet 检测时出错: {e_chardet}。尝试常用编码列表。"
        )

    # 如果 chardet 失败或未启用，尝试常用编码
    if content is None:
        for enc in COMMON_ENCODINGS:
            try:
                async with aiofiles.open(file_path, "r", encoding=enc) as f:
                    content = await f.read()
                logger.info(f"成功使用编码 {enc} 读取文件 {file_path}")
                return content
            except UnicodeDecodeError:
                logger.debug(f"使用编码 {enc} 解码文件 {file_path} 失败")
            except FileNotFoundError:  # 应该在 chardet 步骤就被捕获，但再次检查无妨
                logger.error(f"在尝试常用编码时文件未找到: {file_path}")
                raise
            except Exception as e:
                logger.error(f"使用编码 {enc} 读取文件 {file_path} 时发生错误: {e}")
                # 考虑是否应该 break，如果不是解码错误
                # break

    if content is None:
        logger.error(f"无法使用任何尝试过的编码解码文件 {file_path}")
        # 最后的尝试：使用 utf-8 并替换无法解码的字符
        try:
            logger.warning(
                f"最终尝试：以 UTF-8 编码（替换错误字符）方式读取文件 {file_path}"
            )
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="replace"
            ) as f:
                content = await f.read()
            return content
        except Exception as e_final:
            logger.error(
                f"最终尝试使用 UTF-8 编码读取文件 {file_path}（替换模式）也失败: {e_final}"
            )
            raise ValueError(f"无法读取或解码文件: {file_path}")
    return content


class FileParser:
    """文件解析器主类"""

    def __init__(self, context: Context):
        self.context = context
        self.llm_enabled = False
        self.async_client = None
        self.sync_client = None
        self.model_name = None
        self._setup_llm_clients()

    def _setup_llm_clients(self):
        provider_config = self.context.get_using_provider()
        if provider_config:
            api_key = provider_config.get_current_key()
            api_url = provider_config.provider_config.get("api_base")
            self.model_name = provider_config.get_model()
            if api_key and api_url and self.model_name:
                self.async_client = AsyncOpenAI(api_key=api_key, base_url=api_url)
                self.sync_client = OpenAI(api_key=api_key, base_url=api_url)
                self.llm_enabled = True
                logger.info("FileParser: LLM clients configured successfully.")
            else:
                logger.warning("FileParser: LLM provider config is incomplete. LLM-based parsing will be disabled.")
        else:
            logger.warning("FileParser: No LLM provider configured. LLM-based parsing will be disabled.")

        self.md_converter = MarkItDown(
            enable_plugins=self.llm_enabled,
            llm_client=self.async_client,
            llm_model=self.model_name,
        )

    async def _parse_text(self, file_path: str) -> Optional[str]:
        try:
            return await _detect_and_read_file(file_path)
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            return None

    async def _parse_markdown(self, file_path: str) -> Optional[str]:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self.md_converter.convert(file_path)
            )
            return result.text_content
        except Exception as e:
            logger.error(f"MarkItDown conversion failed for {file_path}: {e}")
            return None

    async def _parse_image(self, file_path: str) -> Optional[str]:
        if not self.llm_enabled:
            logger.warning("LLM parser not enabled, cannot parse image.")
            return None
        try:
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_format = os.path.splitext(file_path)[1].lstrip('.')
            
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "你是图片解析专家，请用当前图片语言提取图片中的文字，只返回纯净的段落文本，不要返回JSON或坐标信息。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}},
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Image parsing failed for {file_path}: {e}")
            return None

    async def _parse_audio(self, file_path: str) -> Optional[str]:
        if not self.llm_enabled:
            logger.warning("LLM parser not enabled, cannot parse audio.")
            return None
        try:
            with open(file_path, "rb") as audio_file:
                base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
            audio_format = os.path.splitext(file_path)[1].lstrip('.')

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "你是音频解析专家，请用当前音频语言提取音频中的文字(中文则使用简体)，只返回纯净的段落文本，不要返回JSON或坐标信息。"},
                            {"type": "input_audio", "input_audio": {"data": base64_audio, "format": audio_format}},
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Audio parsing failed for {file_path}: {e}")
            return None

    async def parse_file_content(self, file_path: str) -> Optional[str]:
        """
        异步读取并解析文件内容。

        Args:
            file_path: 文件路径。

        Returns:
            文件文本内容，如果解析失败则返回 None。
        """
        try:
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()

            # 根据文件类型选择对应的解析器
            if extension in TEXT_EXTENSIONS:
                return await self.text_parser.parse(file_path)
            elif extension in IMAGE_EXTENSIONS:
                return await self.image_parser.parse(file_path)
            elif extension in MARKITDOWN_EXTENSIONS:
                return await self.markdown_parser.parse(file_path)
            elif extension in AUDIO_EXTENSIONS:
                return await self.audio_parser.parse(file_path)
            else:
                logger.warning(f"不支持的文件类型: {extension}，文件路径: {file_path}")
                return None

        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return None
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时发生错误: {e}")
            return None
