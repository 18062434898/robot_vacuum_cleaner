import time

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()
# 当前streamlit的网页是动态的，代码会无时无刻不停地执行，有的一些状态需要保存在session state字典里面去
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
# 需要构建一个历史的message的消息记录
if "message" not in st.session_state:
    st.session_state["message"] = []

# 每一次页面一刷新，整个代码都会从头跑一遍，为了保证以前的对话效果都在页面中有展示，每一次页面重新启动时，
# 需要把message从历史记录里面取出来在页面里面都给他写出去
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    # 用户输入提示词，就把用户的提示词在页面给输出出来
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        # 先不用for循环，先取到他的迭代器对象
        res_stream = st.session_state["agent"].execute_stream(prompt)
        # 传给我这个生成器再原封不动返回出去
        def capture(generator, cache_list):

            for chunk in generator:
                # 把内容往缓存列表里面放，如果没有这段话，本质上这个函数什么都没做
                cache_list.append(chunk)
                # 输出原本时一整段一整段的，现在一个字符一个字符的返回
                for char in chunk:
                    time.sleep(0.01)
                    yield char
        # 把capture拿过来，把res_stream传进去，把缓存response_messages放进去，可以既在页面显示，又在response_messages记录AI返回的内容
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        # 页面一刷新，前面的思考过程就不在了，消息回复完之后手动让页面刷新
        st.rerun()
