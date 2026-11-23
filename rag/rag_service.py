
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService() #向量存储
        self.retriever = self.vector_store.get_retriever()  #检索器
        self.prompt_text = load_rag_prompts()  #提示词文本本身
        self.prompt_template = PromptTemplate.from_template(self.prompt_text) #提示词模板
        self.model = chat_model #需要用到的模型
        self.chain = self._init_chain()  #当前Rag执行的链

    # 构建一个链对象
    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain
    # 检索文档
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)
    # Rag的总结,用户的Query字符串传给我,返回的也是字符串
    def rag_summarize(self, query: str) -> str:
        # 参考资料的文档
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        # document文档拼接
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
