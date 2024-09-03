import re
import os

from pathlib import Path
from urllib.parse import urlparse

from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import FlatReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import QueryBundle
from llama_index.core.schema import MetadataMode

from literature_sentence_window import LiteratureSentenceWindowNodeParser

QA_PROMPT_TMPL_STR = (
    "请你仔细阅读相关内容，结合以下提供的文献资料进行回答。每一条引用的资料都需要标明'出处：《资料名称》原文内容'的形式标注。如果回答需要引用原文，请先给出回答，再贴上对应的原文，使用《资料名称》[]对原文进行标识。如果发现资料无法提供答案，请回答'不知道'。\n"
    "搜索到的相关文献资料如下所示:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "问题: {query_str}\n"
    "答案: "
)

QA_SYSTEM_PROMPT = "你是一个严谨的知识问答智能体，你会仔细阅读提供的文献资料并给出准确的回答。你的回答将非常准确，并且在回答时，会使用《资料名称》[]标识的原文来支持你的回答。如果原文中没有足够的信息来回答问题，你会明确指出这一点。"

REFINE_PROMPT_TMPL_STR = (
    "你是一个知识回答修正机器人，请严格按以下方式工作：\n"
    "1. 只有当原答案为'不知道'时，才进行修正，否则输出原答案的内容。\n"
    "2. 修正时，为了体现你的精准和客观，你会使用《资料名称》[]将原文展示出来。\n"
    "3. 如果感到疑惑时，请用原答案的内容进行回答。\n"
    "新的知识: {context_msg}\n"
    "问题: {query_str}\n"
    "原答案: {existing_answer}\n"
    "新答案: "
)


class Executor:
    def __init__(self):
        pass
    def build_index(selfself,path,overwrite):
        pass
    def build_query_engine(self):
        pass
    def delete_file(self,path):
        pass
    def query(self):
        pass

class MilvusExecutor(Executor):
    def __init__(self, config):
        super().__init__()
        self.index = None
        self.query_engine = None
        self.config = config                #这个传入的config实际上是EasyDict对象
        self.node_parser = LiteratureSentenceWindowNodeParser.from_defaults(
            sentence_splitter=lambda text: re.findall("[^,.;。？！]+[,.;。？！]?", text),
            window_size=config.milvus.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        Settings.llm = OpenAI(temperature=config.llm.temperature, model=config.llm.name, max_tokens=config.llm.max_tokens)      #用于生成回答的大模型
        Settings.embed_model = HuggingFaceEmbedding(model_name=config.embedding.name)                                           #将文本向量化的模型

        self.rerank_postprocessor = SentenceTransformerRerank(model=config.rerank.name, top_n=config.milvus.rerank_topk)        #后处理器，对初次检索的结果rerank，找到与问题文本最相关的topk个，提高检索精度和速度
        self._milvus_client = None
        self._debug = False
    def set_debug(self, mode):
        self._debug = mode

    def build_index(self,path,overwrite):
        config = self.config
        #实例化Milvus数据库对象，与本地的milvus数据集（collection）连接起来
        vector_store = MilvusVectorStore(
            uri=f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name= config.milvus.collection_name,
            overwrite=overwrite,
            dim=config.embedding.dim
        )
        self._milvus_client = vector_store._milvusclient

        #txt-->document
        if path.endswith('.txt'):
            if os.path.exists(path) is False:
                print(f'(rag) 没有找到文件{path}')
                return
            else:
                documents = FlatReader().load_data(Path(path))
                documents[0].metadata['file_name'] = documents[0].metadata['filename']
        elif os.path.isfile(path):
            print('(rag) 目前仅支持txt文件')
        elif os.path.isdir(path):
            if os.path.exists(path) is False:
                print(f'(rag) 没有找到目录{path}')
                return
            else:
                documents = SimpleDirectoryReader(path).load_data()
        else:
            return

        #没啥用，用来给VectorStoreIndex作参数的
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #自己编写的nodeparser，比from_documents这个默认的要好
        nodes = self.node_parser.get_nodes_from_documents(documents)
        #这一步会调用embedding模型进行向量化，
        self.index = VectorStoreIndex(nodes,storage_context=storage_context, show_progress=True)

    def _get_index(self):
        config=self.config
        vector_store = MilvusVectorStore(
            uri=f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name=config.milvus.collection_name,
            dim=config.embedding.dim
        )
        #从现有的向量数据库（里面maybe空的）去构建索引，就比如我之前已经build过了，不想再文本向量化浪费时间
        # ，那么我下次打开就不用再build就能访问上次那个数据库；而此时self.index是空的，所以就需要另外一种方式getindex，调用from_vector_store方法
        # 不过，如果是要把文本向量化的，是必须要用build_index里的方法，要用到storagecontext
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self._milvus_client = vector_store._milvusclient

    def build_query_engine(self):
        config = self.config
        if self.index is None:
            self._get_index()
        self.query_engine = self.index.as_query_engine(
            #添加查询引擎中的后处理器
            node_postprocessors=[
                self.rerank_postprocessor,
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ]
        )
        self.query_engine.retriever.similarity_top_k=config.milvus.retrieve_topk

        message_templates = [
            ChatMessage(content=QA_SYSTEM_PROMPT,role=MessageRole.SYSTEM),   # 系统消息模板，描述系统在对话中的身份和作用
            ChatMessage(content=QA_PROMPT_TMPL_STR,role=MessageRole.USER)   # 用户消息模板，对用户的输入问题所需回答的补充
        ]
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        #更新查询引擎中的提示模板
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template":chat_template}
        )
        self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR

    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = \
        self._milvus_client.query(collection_name='history_rag', filter="", output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(collection_name=config.milvus.collection_name, filter=f"file_name=='{path}'")
        num_entities = \
        self._milvus_client.query(collection_name='history_rag', filter="", output_fields=["count(*)"])[0]["count(*)"]
        print(f'(rag) 现有{num_entities}条，删除{num_entities_prev - num_entities}条数据')

    def query(self,question):
        if self.index is None:
            self._get_index()
        if question.endswith('?') or question.endswith('？'):    #避免问号？干扰问题文本与参考资料的相似度
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))    #获取与查询到的问题文本相关的上下文内容
            for i,context in enumerate(contexts):
                print(f'{question}',i)
                content = context.node.get_content(metadata_mode=MetadataMode.LLM)
                print(content)
            print('-----------------------------------------------------------参考资料------------------------------------------------------------')
        response = self.query_engine.query(question)
        return response