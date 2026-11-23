from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),  # 系统提示词
            # 工具
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            # messages是key，value是一个列表，列表中每一条消息都是一个字典，字典中有role和content两个key
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记，先要给它一个初始化值，report为false，默认为false
        for chunk in self.agent.stream(
            input_dict, stream_mode="values", context={"report": False}
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"  # 把流的消息也通过流的方式对外提供，也就是通过yield表达式


if __name__ == "__main__":
    agent = ReactAgent()

    # for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
    #     print(chunk, end="", flush=True)
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
