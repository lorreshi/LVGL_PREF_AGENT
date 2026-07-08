"""具体 Retriever 后端子包。

LocalStoreRetriever（基于 LangGraph Store 的语义召回，默认，任务 5.2）与
ExternalRAGRetriever（以 tool 形式调用外部 RAG 的骨架，任务 5.2）由本子包
实现，并经 ``@RetrieverFactory.register(...)`` 注册。导入本子包应触发各后端
注册。
"""
