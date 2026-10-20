import json
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.base import setup_logging


def router(state: ImportGraphState) -> str:

    if state.get('is_md_read_enabled'):
        return "md"
    if state.get('is_pdf_read_enabled'):
        return "pdf"
    # raise ValueError("is_md_read_enabled或is_pdf_read_enabled必须有一个是true")
    return END  # 安全降级

"""
    创建导入流程图

    Returns:
        编译后的 StateGraph 实例

    流程结构:
        entry_node
              │
              ├── (PDF) ──> pdf_to_md_node ──┐
              │                              │
              └── (MD) ─────────────────────>├──> md_img_node
                                              │
                                              v
                                      document_split_node
                                              │
                                              v
                                      item_name_rec_node
                                              │
                                              v
                                        bge_embedding_node
                                              │
                                              v
                                        import_milvus_node
                                              │
                                              v
                                             END
    """
def create_import_graph() -> CompiledStateGraph:

    # 1. 定义状态图
    graph = StateGraph(ImportGraphState)

    # 2. 定义节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    # 2.1 添加入口节点
    graph.set_entry_point("entry_node")

    # 2.2 添加所有节点到图中
    for key, value in nodes.items():
        graph.add_node(key, value)

    # 3. 定义边（条件边、顺序边）
    # 3.1 条件边：入口节点后根据文件类型路由
    # 定义路由映射表（routing map），将路由函数的返回值映射到实际的下一个节点。
    # 键（Key）：router 函数可能返回的值
    # 值（Value）：实际要跳转到的目标节点名称
    graph.add_conditional_edges(
        "entry_node",
        router,
        {
            "md": "md_img_node",
            "pdf": "pdf_to_md_node",
            END: END
        }
    )

    # 3.2 顺序边
    graph.add_edge("pdf_to_md_node", "md_img_node")
    graph.add_edge("md_img_node", "document_split_node")
    graph.add_edge("document_split_node", "item_name_recognition_node")
    graph.add_edge("item_name_recognition_node", "bge_embedding_node")
    graph.add_edge("bge_embedding_node", "import_milvus_node")
    graph.add_edge("import_milvus_node", END)

    # 4. 编译图
    return graph.compile()


# 创建全局图实例
kb_import_graph = create_import_graph()

def run_import_graph(import_file_path: str, file_dir: str) -> dict:
    """
    便捷函数：运行导入流程

    Args:
        import_file_path: 输入文件路径（PDF 或 MD）
        file_dir: 本地工作目录

    Returns:
        最终状态字典
    """
    # 1. 构建初始状态
    state = {
        "is_md_read_enabled": False,
        "is_pdf_read_enabled": True,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    init_state = create_default_state(**state)

    # 2. 调用 stream 获取每个节点的处理结果
    final_state = None
    for event in kb_import_graph.stream(init_state):
        for node_name, state in event.items():
            print(f"运行节点: {node_name}")
            final_state = state

    return final_state


if __name__ == "__main__":
    setup_logging()

    import_file_path = r"D:\doc\万用表RS-12的使用.pdf"
    file_dir = r"D:\temp_dir"

    # 1. 测试编排流程
    final_state = run_import_graph(import_file_path,file_dir)
    print(json.dumps(final_state, indent=4, ensure_ascii=False))

    # 2. 打印图结构（ASCII 可视化）
    print("-" * 50)
    print("图结构:")
    kb_import_graph.get_graph().print_ascii()