from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, CSVLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

class DocumentLoader:
    """Classe responsável por carregar documentos de diferentes formatos."""
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()

    @staticmethod
    def load_csv(file_path: str) -> List[Document]:
        loader = CSVLoader(file_path)
        docs = loader.load()
        return docs

    @staticmethod
    def load_json(file_path: str) -> List[Document]:
        loader = JSONLoader(file_path, jq_schema=".[]", text_content=False)
        return loader.load()

    @staticmethod
    def load_web(url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(loader.load())
       

    @staticmethod
    def load_youtube(url: str) -> List[Document]:
        # Extrair o ID do vídeo da URL
        video_id = url.split("v=")[-1].split("&")[0]
        
        # Obter informações do vídeo usando yt-dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', '')
                description = info.get('description', '')
            except Exception as e:
                print(f"Erro ao obter informações do vídeo: {e}")
                title = ""
                description = ""
        
        # Obter a transcrição
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt'])
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript)
        except Exception as e:
            print(f"Erro ao obter transcrição: {e}")
            transcript_text = ""
        
        # Combinar todas as informações
        content = f"Título: {title}\n\nDescrição: {description}\n\nTranscrição:\n{transcript_text}"
        
        return [Document(page_content=content)]

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        docs = loader.load()
        doc = '\n\n'.join([d.page_content for d in docs])
        return doc

    def load_document(self, file_path: str) -> List[Document]:
        """Carrega documento com base na extensão do arquivo."""
        if file_path.endswith(".pdf"):
            return self.load_pdf(file_path)
        elif file_path.endswith(".csv"):
            return self.load_csv(file_path)
        elif file_path.endswith(".json"):
            return self.load_json(file_path)
        elif file_path.endswith(".txt"):
            return self.load_text(file_path)
        else:
            raise ValueError("Formato de arquivo não suportado")
        