# langchain-rag

Este é um projeto de Retrieval-Augmented Generation (RAG) que utiliza a tecnologia de IA para responder perguntas sobre o conteúdo do site da Nearx, utilizando OpenAI como provedores de LLM (Large Language Models).

## 🚀 Funcionalidades

- Carregamento automático de conteúdo web
- Processamento de texto usando embeddings
- Armazenamento vetorial com ChromaDB
- Suporte ao Modelos de linguagem (OpenAI)
- Sistema de memória para manter contexto das conversas
- Streaming de respostas em tempo real
- Monitoramento e debug com LangSmith
- Rastreamento de execuções e métricas de performance
- Análise detalhada de cada etapa do processo RAG

## 📋 Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Chaves de API (OpenAI)
- Conta no LangSmith (para monitoramento)

## 🔧 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/langchain-rag.git
cd langchain-rag
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure o arquivo `.env`:
```env
OPENAI_API_KEY=sua_chave_openai
LANGSMITH_API_KEY=sua_chave_langsmith
LANGSMITH_TRACING=true
```

## 🎮 Como Usar

1. Execute o script principal:
```bash
python main.py
```

2. Digite suas perguntas quando solicitado. O sistema irá:
   - Buscar informações relevantes no conteúdo armazenado
   - Gerar respostas contextualizadas
   - Manter histórico da conversa

## 🛠️ Estrutura do Projeto

```
langchain-rag/
├── src/
│   ├── document_loader.py
│   ├── embedding_manager.py
│   ├── rag_chain.py
│   └── utils.py
├── .env
├── main.py
└── requirements.txt
```

## 🔑 Variáveis de Ambiente

O arquivo `.env` deve conter:

- `OPENAI_API_KEY`: Chave de API da OpenAI
- `LANGSMITH_API_KEY`: Chave de API do LangSmith (opcional)
- `LANGSMITH_TRACING`: Habilita rastreamento do LangSmith


## 📚 Tecnologias Utilizadas

- LangChain
- ChromaDB
- OpenAI
- LangSmith
- Python

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## ✨ Agradecimentos

- Equipe Nearx
- Comunidade LangChain
- Contribuidores OpenAI

## 🔍 Monitoramento com LangSmith

O projeto utiliza o LangSmith para:
- Monitorar a performance dos modelos
- Rastrear cada etapa do processo RAG
- Debugar e otimizar as chains
- Coletar métricas de uso
- Avaliar a qualidade das respostas

Para habilitar o LangSmith:
1. Crie uma conta em https://smith.langchain.com/
2. Obtenha sua chave de API
3. Configure as variáveis de ambiente no `.env`:
```env
LANGSMITH_API_KEY=sua_chave_langsmith
LANGSMITH_TRACING=true
``` 