#参考文档：https://www.milvus-io.com/integrations/integrate_with_llamaindex

#1、配置Setting（准备api key）
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#用于交互的大模型，OPENAI()方法会自动调用环境变量里的OPENAI_API_KEY
Settings.llm=OpenAI(temperature=0.01, model="gpt-4o-2024-08-06", max_tokens=2048)
#用于文本向量化的embedding模型，VectorStoreIndex自动调用
Settings.embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")



#2、加载文档，获取文档id(Doc_id)，使其对llamaindex可访问
#如果是更大一些的系统，会再需要将文档解析为节点，则要再写一个nodeparser类
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import FlatReader
from pathlib import Path

#SimpleDirectoryReader和FlatReader分别是读文件夹、文件的；使用方法也不同；但是返回值都是Document对象列表，
#documents = SimpleDirectoryReader("文件夹目录").load_data()
#FlatReader().load_data(Path("文件路径"))   ，只接受Path()对象，不接受str
documents = FlatReader().load_data(Path("./test.txt"))
print(documents[0].doc_id)



#3、创建一个Milvus集合(collection)，并将上一步的文档(Document)插入其中

from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext,VectorStoreIndex

#vector_store是llamaindex库与Milvus数据库交互的实例，借助MilvusVectorStore类的方法
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="history_rag",
    overwrite=True,
    dim=768
)
#StorageContext是一个上下文管理器，from_defaults()方法接受一个xxxVectorStore类型的实例
#这里就是与Milvus数据库的实例（该集合名为collection_name）建立了连接
storage_context = StorageContext.from_defaults(vector_store=vector_store)
#VectorStoreIndex负责将数据向量化，并存储到数据库中
#必须接收一个StorageContext变量（这样才能知道把向量数据存到哪里，且能够进行上下文管理）
#这里调用的是from_documents方法，所以第一个参数是第一部中载入的文档实例，会隐式地将文档转换为节点
#也可以自己写一个nodeparser，然后直接调用VectorStoreIndex(nodes, storage_context=storage_context,...)
index = VectorStoreIndex.from_documents(documents,storage_context=storage_context,show_progress=True)

#大体的连接逻辑就是：txt --(FlatReader)--> document--> Index <--Storage_context<--MilvusVectorStore
#Index就是连接了文档和数据库的桥梁


#4、查询数据
#此前已经将txt数据向量化并存储在了Milvus数据库中，llamaindex将使用该集合中的数据库作为GPT生成答案的知识库
#index的as_query_engine()方法可以通过llamaindex，让gpt用数据集中的内容去回答，返回值是回答的内容

#初始化查询引擎
import textwrap
query_engine = index.as_query_engine()

while True:
    #获取用户输入
    question = input("请输入您的问题（输入'退出'结束）：")

    # 如果用户输入'退出'，则结束循环
    if question.lower() == '退出':
        break

    # 执行查询并打印结果
    response = query_engine.query(question)
    print(textwrap.fill(str(response), 100))