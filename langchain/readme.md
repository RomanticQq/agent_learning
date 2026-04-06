RAG->langchain
agent->langgraph

现在
langchain既可以做RAG，也可以做agent。但是要做高度定制的agent，还是需要做langgraph


模型包装器
数据增强：RAG
记忆：一般在多轮对话中会用到
agent：
链：组件之间的链接

langchain v1.x主要新增内容（相较于v0.3）

中间件：人工审核的中间件，自定义装饰器中间件、自定义类中间件

标准化输出 content_block

动态模型中间件