from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call,
    before_model,
    dynamic_prompt,
    ModelRequest,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
    # 请求的数据封装,可以看作入参
    request: ToolCallRequest,
    # 执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:  # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)  # 拿到handle,也就是函数，用这个函数去给它传参
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        # 如果调用了这个工具，这边就可以监控到它
        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
    state: AgentState,  # 整个Agent智能体中的状态记录
    runtime: Runtime,  # 记录了整个执行过程中的上下文信息
):  # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    # 级别设置为debug才显示，级别不是debug，是高于debug就不显示了，
    # 上面这个日志知道有多少条消息调用进去就可以，消息本身并不一定完全是必须的
    logger.debug(
        f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}"
    )

    return None


# main提示词回答用户然后跟用户聊天，用户需要生成报告，模型识别到需求就调用这个工具
@dynamic_prompt  # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):  # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:  # 如果是true,是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
