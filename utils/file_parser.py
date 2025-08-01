from typing import Optional
from astrbot.api import logger
from astrbot.api.star import Context
import os
import base64
import aiofiles
from aiofiles.os import stat as aio_stat
import chardet
import asyncio
from ..config.settings import LLMSettings
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

    def __init__(self, context: Context, llm_settings: LLMSettings):
        self.context = context
        self.llm_settings = llm_settings
        self.llm_enabled = False
        self.async_client = None
        self.sync_client = None
        self.model_name = None
        # 延迟初始化，等到实际需要使用时再设置LLM客户端
        self.md_converter = None
        self._llm_setup_attempted = False

    async def _setup_llm_clients(self):
        """异步设置LLM客户端"""
        if self._llm_setup_attempted:
            return  # 避免重复设置
        
        self._llm_setup_attempted = True
        
        if not self.llm_settings.enable_llm_parser:
            logger.info("FileParser: LLM解析功能已禁用。")
            self.llm_enabled = False
            self._setup_markitdown()
            return

        provider_config = None
        provider_id = self.llm_settings.provider

        if not provider_id:
            logger.info("FileParser: 未指定LLM提供商ID，将尝试使用AstrBot的默认提供商。")
            provider_config = self.context.get_using_provider()
            if not provider_config:
                logger.warning("FileParser: 当前没有正在使用的LLM提供商。")
        else:
            logger.info(f"FileParser: 尝试使用指定的LLM提供商ID: {provider_id}")
            provider_config = self.context.get_provider_by_id(provider_id)
            if not provider_config:
                logger.warning(f"FileParser: 未找到ID为 '{provider_id}' 的LLM提供商。")

        if not provider_config:
            # 如果没有找到指定提供商，列出所有可用的提供商供参考
            all_providers = self.context.get_all_providers()
            if all_providers:
                provider_ids = [p.provider_id for p in all_providers]
                logger.info(f"FileParser: 可用的LLM提供商ID列表: {provider_ids}")
            else:
                logger.warning("FileParser: 系统中没有配置任何LLM提供商。")
            
            logger.warning("FileParser: 无法获取LLM提供商配置，基于LLM的解析将被禁用。")
            self.llm_enabled = False
            self._setup_markitdown()
            return

        try:
            # 获取提供商配置信息
            api_key = provider_config.get_current_key()
            api_url = provider_config.provider_config.get("api_base")
            
            # 获取模型信息：异步调用get_models()
            try:
                available_models = await provider_config.get_models()
                if available_models:
                    # 使用配置中指定的模型，或者使用第一个可用模型
                    self.model_name = provider_config.provider_config.get("model")
                    if not self.model_name or self.model_name not in available_models:
                        self.model_name = available_models[0]
                        logger.info(f"FileParser: 使用默认模型: {self.model_name}")
                else:
                    logger.warning("FileParser: 提供商没有可用的模型列表。")
                    self.model_name = provider_config.provider_config.get("model", "gpt-3.5-turbo")
                    logger.info(f"FileParser: 使用配置中的模型: {self.model_name}")
            except Exception as e:
                logger.warning(f"FileParser: 异步获取模型列表失败: {e}")
                self.model_name = provider_config.provider_config.get("model", "gpt-3.5-turbo")
                logger.info(f"FileParser: 使用配置中的模型: {self.model_name}")
            
            if not api_key:
                logger.warning("FileParser: LLM提供商API密钥为空。")
            if not api_url:
                logger.warning("FileParser: LLM提供商API地址为空。")
            if not self.model_name:
                logger.warning("FileParser: LLM提供商模型名称为空。")
            
            if api_key and api_url and self.model_name:
                self.async_client = AsyncOpenAI(api_key=api_key, base_url=api_url)
                self.sync_client = OpenAI(api_key=api_key, base_url=api_url)
                self.llm_enabled = True
                
                # 获取provider ID，使用meta()方法
                try:
                    provider_id = provider_config.meta().id
                except Exception:
                    provider_id = "unknown"
                    
                logger.info(
                    f"FileParser: LLM客户端配置成功 (Provider: {provider_id}, Model: {self.model_name}, API: {api_url})。"
                )
            else:
                logger.warning("FileParser: LLM提供商配置不完整，基于LLM的解析将被禁用。")
                self.llm_enabled = False
                
        except Exception as e:
            logger.error(f"FileParser: 获取LLM提供商配置时出错: {e}，基于LLM的解析将被禁用。", exc_info=True)
            self.llm_enabled = False

        self._setup_markitdown()

    def _setup_markitdown(self):
        """设置MarkItDown转换器"""
        try:
            if self.llm_enabled:
                self.md_converter = MarkItDown(
                    enable_plugins=True,
                    llm_client=self.async_client,
                    llm_model=self.model_name,
                )
                logger.info("FileParser: MarkItDown初始化成功，启用LLM插件。")
            else:
                self.md_converter = MarkItDown(enable_plugins=False)
                logger.info("FileParser: MarkItDown初始化成功，禁用LLM插件。")
        except Exception as e:
            logger.error(f"FileParser: MarkItDown初始化失败: {e}，将影响复杂文件解析功能。", exc_info=True)
            # 即使MarkItDown初始化失败，也要创建一个基础实例
            try:
                self.md_converter = MarkItDown(enable_plugins=False)
                logger.info("FileParser: 使用基础MarkItDown配置。")
            except Exception as e2:
                logger.error(f"FileParser: 基础MarkItDown初始化也失败: {e2}")
                self.md_converter = None

    async def _parse_text(self, file_path: str) -> Optional[str]:
        try:
            return await _detect_and_read_file(file_path)
        except Exception as e:
            logger.error(f"解析文本文件 {file_path} 时出错: {e}")
            return None

    async def _parse_markdown(self, file_path: str) -> Optional[str]:
        try:
            if not self.md_converter:
                logger.warning(f"MarkItDown未正确初始化，无法解析文件: {file_path}")
                # 尝试作为普通文本读取
                return await _detect_and_read_file(file_path)
                
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self.md_converter.convert(file_path)
            )
            return result.text_content
        except Exception as e:
            logger.error(f"MarkItDown转换失败: {file_path}: {e}")
            # 回退到普通文本读取
            try:
                logger.info(f"尝试作为普通文本读取文件: {file_path}")
                return await _detect_and_read_file(file_path)
            except Exception as e2:
                logger.error(f"作为普通文本读取也失败: {file_path}: {e2}")
                return None

    async def _parse_image(self, file_path: str) -> Optional[str]:
        if not self.llm_enabled:
            logger.warning("LLM解析器未启用，无法解析图像。")
            return None
        try:
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_format = os.path.splitext(file_path)[1].lstrip(".")

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "你是图片解析专家，请用当前图片语言提取图片中的文字，只返回纯净的段落文本，不要返回JSON或坐标信息。",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"图像解析失败: {file_path}: {e}")
            return None

    async def _parse_audio(self, file_path: str) -> Optional[str]:
        if not self.llm_enabled:
            logger.warning("LLM解析器未启用，无法解析音频。")
            return None
        try:
            with open(file_path, "rb") as audio_file:
                base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
            audio_format = os.path.splitext(file_path)[1].lstrip(".")

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "你是音频解析专家，请用当前音频语言提取音频中的文字(中文则使用简体)，只返回纯净的段落文本，不要返回JSON或坐标信息。",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64_audio,
                                    "format": audio_format,
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"音频解析失败: {file_path}: {e}")
            return None

    async def parse_file_content(self, file_path: str) -> Optional[str]:
        """
        异步读取并解析文件内容。

        Args:
            file_path: 文件路径。

        Returns:
            文件文本内容，如果解析失败则返回 None。
        """
        # 确保LLM客户端已经设置
        if not self._llm_setup_attempted:
            await self._setup_llm_clients()
            
        try:
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()

            # 根据文件类型选择对应的解析器
            if extension in TEXT_EXTENSIONS:
                return await self._parse_text(file_path)
            elif extension in IMAGE_EXTENSIONS:
                return await self._parse_image(file_path)
            elif extension in MARKITDOWN_EXTENSIONS:
                return await self._parse_markdown(file_path)
            elif extension in AUDIO_EXTENSIONS:
                return await self._parse_audio(file_path)
            else:
                logger.warning(f"不支持的文件类型: {extension}，文件路径: {file_path}")
                return None

        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return None
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时发生错误: {e}")
            return None
