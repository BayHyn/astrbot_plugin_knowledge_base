from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitterUtil:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initializes the text splitter using RecursiveCharacterTextSplitter from langchain.
        
        Args:
            chunk_size: The target size of each chunk (in characters).
            chunk_overlap: The size of the overlap between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", "。", "，", ". ", ", ", " ", ""],
        )

    def split_text(self, text: str) -> List[str]:
        """
        Splits the text into chunks.
        
        Args:
            text: The text to be split.
            
        Returns:
            A list of text chunks.
        """
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text)
