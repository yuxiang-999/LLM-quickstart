from langchain.chains import ConversationChain, TransformChain, LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI


llm = OpenAI()


# 定义一个转换函数，输入是一个字典，输出也是一个字典。
def transform_func(inputs: dict) -> dict:
    # 从输入字典中获取"text"键对应的文本。
    text = inputs["text"]
    if len(text) > 1000:
        text = text[:1000] + " ......"
    # 返回问题的文本，用"input"作为键。
    return {"input": text}


def create_transform_chain() -> TransformChain:
    # 使用上述转换函数创建一个TransformChain对象。
    # 定义输入变量为["text"]，输出变量为["input"]，并指定转换函数为transform_func。
    transform_chain = TransformChain(
        input_variables=["text"], output_variables=["input"], transform=transform_func
    )
    return transform_chain


from langchain.prompts import PromptTemplate


def create_destination_chains():

    summary_template = """
    你是一位卓越的文本摘要专家。
    你擅长从复杂的文本中提炼核心信息，生成简明扼要的摘要。
    当你面对某些不确定或难以理解的内容时，你会以透明的方式表达你的困惑。
    
    请根据下面内容编写文本摘要：
    {input}
    """

    translation_template = """
    你是一位优秀的翻译专家。
    你擅长以清晰简洁的方式把英文翻译成中文，让内容易于理解。
    当你遇到某些不确定或难以准确表达的内容时，你会诚实地表达你的疑惑。
    
    请把下面英文内容翻译成中文：
    {input}
    """

    prompt_infos = [
        {
            "name": "文本摘要",
            "description": "适用于回答文本摘要问题",
            "prompt_template": summary_template,
        },
        {
            "name": "英文翻译",
            "description": "适用于回答英文翻译问题",
            "prompt_template": translation_template,
        },
    ]

    # 创建一个空的目标链字典，用于存放根据prompt_infos生成的LLMChain。
    destination_chains = {}

    # 遍历prompt_infos列表，为每个信息创建一个LLMChain。
    for p_info in prompt_infos:
        name = p_info["name"]  # 提取名称
        prompt_template = p_info["prompt_template"]  # 提取模板
        # 创建PromptTemplate对象
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        # 使用上述模板和llm对象创建LLMChain对象
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        # 将新创建的chain对象添加到destination_chains字典中
        destination_chains[name] = chain

    return prompt_infos, destination_chains


from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain


def create_multi_prompt_chain() -> MultiPromptChain:
    # 创建目标链
    prompt_infos, destination_chains = create_destination_chains()
    # 从prompt_infos中提取目标信息并将其转化为字符串列表
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    # 使用join方法将列表转化为字符串，每个元素之间用换行符分隔
    destinations_str = "\n".join(destinations)
    # 根据MULTI_PROMPT_ROUTER_TEMPLATE格式化字符串和destinations_str创建路由模板
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    # 创建路由的PromptTemplate
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    # 使用上述路由模板和llm对象创建LLMRouterChain对象
    router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt, verbose=True)

    # 创建一个默认的ConversationChain
    default_chain = ConversationChain(llm=llm, output_key="text", verbose=True)

    # 创建MultiPromptChain对象，其中包含了路由链，目标链和默认链。
    multi_prompt_chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    return multi_prompt_chain


def create_sequential_chain() -> SimpleSequentialChain:
    transform_chain = create_transform_chain()
    multi_prompt_chain = create_multi_prompt_chain()
    sequential_chain = SimpleSequentialChain(chains=[transform_chain, multi_prompt_chain])
    return sequential_chain


if __name__ == "__main__":
    sequential_chain = create_sequential_chain()

    summary_question = """
    请根据下面内容编写文本摘要：
    当地时间2月13日，OpenAI公司宣布，正在测试ChatGPT的记忆能力。ChatGPT将记住用户在所有聊天中讨论过的事情，这可以避免重复信息，
    使未来的对话更有帮助。用户可以控制ChatGPT的记忆，可以明确告诉它记住某些东西，询问它记住了什么，也可以通过对话或设置告诉它忘记，
    还可以完全关闭记忆功能。OpenAI称，本周将向一小部分ChatGPT免费和Plus用户推出该功能。
    """

    translation_question = """
    请把下面英文内容翻译成中文：
    In the last three years, the largest dense deep learning models have grown over 1000x to reach hundreds of billions 
    of parameters, while the GPU memory has only grown by 5x (16 GB to 80 GB). Therefore, the growth in model scale has 
    been supported primarily though system innovations that allow large models to fit in the aggregate GPU memory of 
    multiple GPUs. However, we are getting close to the GPU memory wall. It requires 800 NVIDIA V100 GPUs just to fit 
    a trillion parameter model for training, and such clusters are simply out of reach for most data scientists. 
    In addition, training models at that scale requires complex combinations of parallelism techniques that puts a big 
    burden on the data scientists to refactor their model.
    """

    response = sequential_chain.run(summary_question)

    # response = sequential_chain.run(translation_question)

    print(response)
