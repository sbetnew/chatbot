import openai
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
import pandas as pd

# Inicializar OpenAI LLM (definir sua chave de API OpenAI)
openai.api_key = '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'
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-4"))

# Função para carregar os produtos de um arquivo CSV
def carregar_produtos_csv(caminho_arquivo):
    produtos = pd.read_csv(caminho_arquivo)
    return produtos

# Função para criar o índice do LlamaIndex com base nos produtos
def criar_indice_produtos(produtos):
    documentos = []
    for _, row in produtos.iterrows():
        documento = f"Produto: {row['nome']}. Descrição: {row['descricao']}. Preço: {row['preco']} R$. Estoque: {row['estoque']} unidades."
        documentos.append(documento)
    return GPTSimpleVectorIndex.from_documents(documentos, llm_predictor)

# Carregar os produtos e criar o índice
produtos = carregar_produtos_csv('produtos.csv')
indice = criar_indice_produtos(produtos)