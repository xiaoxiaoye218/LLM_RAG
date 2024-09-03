"""Customize Simple node parser."""
from typing import Any, Callable, List, Optional, Sequence
from bisect import bisect_right

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core import Document
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_WINDOW_SIZE = 3                         #窗口大小决定了在句子前后捕获多少句子作为上下文信息
DEFAULT_WINDOW_METADATA_KEY = "window"          #指定存储窗口元数据的键名。 在解析文本并生成节点时，每个节点会保存它的上下文窗口信息，这些信息会被存储在节点的元数据中，窗口信息的键名就是“window”
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"  #指定存储原始句子的元数据键名。 在生成节点时，每个节点还会保存其对应的原始句子内容，这些内容也会被存储在节点的元数据中，原始内容的键名就是“original_text"


class LiteratureSentenceWindowNodeParser(NodeParser):
    # 当子类没有__init__()方法时，会自动调用父类的__init__()方法

    #python的类型注解和llama.index的Field方法
    #Callable[]关键字说明是一个这是一个函数,这里是接受一个str，返回一个List[str]
    #Callable[[arg1_type, arg2_type, ...], return_type]
    #Field方法：指定字段的默认值、并增加一些描述
    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )
    window_size: int = Field(
        default=DEFAULT_WINDOW_SIZE,
        description="The number of sentences on each side of a sentence to capture.",
        gt=0,
    )
    window_metadata_key: str = Field(
        default=DEFAULT_WINDOW_METADATA_KEY,
        description="The metadata key to store the sentence window under.",
    )
    original_text_metadata_key: str = Field(
        default=DEFAULT_OG_TEXT_METADATA_KEY,
        description="The metadata key to store the original sentence in.",
    )

    # 装饰器，用于将一个方法转换为类方法，可以通过类名调用，可以修改类中的变量
    @classmethod
    def book_name(cls, path):
        #建立一个文件名---汉字书名的映射
        _mapping = {}
        _mapping["baihuabeiqishu"] = "北齐书"
        _mapping["baihuabeishi.txt"] = "北史"
        _mapping["baihuachenshu.txt"] = "陈书"
        _mapping["baihuahanshu.txt"] = "汉书"
        _mapping["baihuahouhanshu.txt"] = "后汉书"
        _mapping["baihuajinshi.txt"] = "金史"
        _mapping["baihuajinshu.txt"] = "晋书"
        _mapping["baihuajiutangshu.txt"] = "旧唐书"
        _mapping["baihuajiuwudaishi.txt"] = "旧五代史"
        _mapping["baihualiangshu.txt"] = "梁书"
        _mapping["baihualiaoshi.txt"] = "辽史"
        _mapping["baihuamingshi.txt"] = "明史"
        _mapping["baihuananqishu.txt"] = "南齐书"
        _mapping["baihuananshi.txt"] = "南史"
        _mapping["baihuasanguozhi.txt"] = "三国志"
        _mapping["baihuashiji.txt"] = "史记"
        _mapping["baihuasongshi.txt"] = "宋史"
        _mapping["baihuasongshu.txt"] = "宋书"
        _mapping["baihuasuishu.txt"] = "隋史"
        _mapping["baihuaweishu.txt"] = "魏书"
        _mapping["baihuaxintangshi.txt"] = "新唐史"
        _mapping["baihuaxinwudaishi.txt"] = "新五代史"
        _mapping["baihuayuanshi.txt"] = "元史"
        _mapping["baihuazhoushu.txt"] = "周书"

        #如果遍历到文件名在映射里，那么就返回名字，否则返回未名
        for name in _mapping:
            if name in path:
                return _mapping[name]
        return "未名"

    #创建一个带有默认参数的LiteratureSentenceWindowNodeParser
    @classmethod
    def from_defaults(
        cls,                #类方法的默认参数是cls，而不是self
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        window_metadata_key: str = DEFAULT_WINDOW_METADATA_KEY,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,          #是否在生成的节点中包含文档的元数据
        include_prev_next_rel: bool = True,     #是否在生成的节点中包含与前后句子的关联信息
        callback_manager: Optional[CallbackManager] = None,
    ) -> "LiteratureSentenceWindowNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()

        return cls(
            sentence_splitter=sentence_splitter,
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    #将文档解析为适合向量化处理的节点
    #自动覆盖父类的同名方法
    def _parse_nodes(
        self,  # 类的实例自身
        nodes: Sequence[BaseNode],  # 传入的节点序列，类型为 BaseNode 的序列
        show_progress: bool = False,  # 是否显示进度条，默认为 False
        **kwargs: Any,  # 其他可选参数
    ) -> List[BaseNode]:  # 返回值类型是 BaseNode 的列表
        """Parse document into nodes."""

        # 初始化一个空列表，用于存储所有解析后的节点
        all_nodes: List[BaseNode] = []

        # 如果 show_progress 为 True，显示进度条，并开始遍历传入的节点
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            # 对每个节点的内容进行句子分割，不对结果进行任何处理，仅用于切割
            self.sentence_splitter(node.get_content(metadata_mode=MetadataMode.NONE))

            # 调用 build_window_nodes_from_documents 函数，将每个节点处理为多个节点
            nodes = self.build_window_nodes_from_documents([node])

            # 将处理后的节点扩展添加到 all_nodes 列表中
            all_nodes.extend(nodes)

        # 返回所有处理后的节点
        return all_nodes

    def build_window_nodes_from_documents(
        self,  # 类的实例自身
        documents: Sequence[Document]  # 传入的文档序列
    ) -> List[BaseNode]:  # 返回值类型是 BaseNode 的列表
        """Build window nodes from documents."""

        # 初始化一个空列表，用于存储所有解析后的节点
        all_nodes: List[BaseNode] = []
        # 遍历每个文档
        for doc in documents:
            text = doc.text  # 获取文档内容
            title_localizer = self.analyze_titles(text)  # 分析标题并获取一个 TitleLocalizer 对象
            lines = text.split('\n')  # 按行分割文档内容
            nodes = []  # 初始化一个空列表，用于存储处理后的节点
            book_name = LiteratureSentenceWindowNodeParser.book_name(doc.metadata['file_name'])  # 获取文档的书名

            # 遍历每一行文本
            for i, line in enumerate(lines):
                if len(line) == 0:
                    continue
                # 对每一行进行句子分割
                text_splits = self.sentence_splitter(line)

                # 构建节点，将分割的句子转换为节点对象
                line_nodes = build_nodes_from_splits(
                    text_splits,
                    doc,
                    id_func=self.id_func,
                )
                # 获取当前行对应的标题
                title = title_localizer.get_title_line(i)
                if title == None:
                    continue
                # 给每个节点添加出处元数据
                for line_node in line_nodes:
                    line_node.metadata["出处"] = f"《{book_name}·{title[0]}》"
                nodes.extend(line_nodes)

            # 为每个节点添加上下文窗口的元数据
            for i, node in enumerate(nodes):
                #前后各捕获3句
                window_nodes = nodes[
                    max(0, i - self.window_size) : min(i + self.window_size, len(nodes))
                ]
                #将window_nodes中的每个节点的text文本 拼接成一个字符串，并将其存储在当前节点的窗口元数据中
                node.metadata[self.window_metadata_key] = " ".join(
                    [n.text for n in window_nodes]
                )
                #将当前节点的原始文本存储在节点的元数据中
                node.metadata[self.original_text_metadata_key] = node.text

                #排除无关字段
                node.excluded_embed_metadata_keys.extend(
                    [self.window_metadata_key, self.original_text_metadata_key, 'title', 'file_path', '出处', 'file_name', 'filename', 'extension']
                )
                node.excluded_llm_metadata_keys.extend(
                    [self.window_metadata_key, self.original_text_metadata_key, 'file_path', 'file_name', 'filename', 'extension']
                )

                all_nodes.append(node)
        return all_nodes


    #暴力遍历每一行，获取标题的所在行数;但是是通过对标题的文本内容来分析的，具有一定的局限性，对历史文献效果最佳
    def analyze_titles(self, text):
        lines = text.split('\n')
        titles = []
        for i, line in enumerate(lines):
            if len(line) > 0 and line[0] != '\n' and line[0] != '\u3000' and line[0] != ' ':
                if '纪' not in line and '传' not in line:
                    continue
                titles.append([line.strip(), i])
        return TitleLocalizer(titles, len(lines))

#根据标题所在的行号，进行二分查找line_id在哪个标题的管辖行中
class TitleLocalizer():
    def __init__(self, titles, total_lines):
        self._titles = titles
        self._total_lines = total_lines

    def get_title_line(self, line_id):
        indices = [title[1] for title in self._titles]
        index = bisect_right(indices, line_id)
        if index - 1 < 0:
            return None
        return self._titles[index-1]


