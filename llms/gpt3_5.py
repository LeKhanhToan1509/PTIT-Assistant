from langchain_openai import ChatOpenAI

class GPT3_5():
    def __init__(self, llm= None, api_key = None, temperature = 0.5):
        self.temperature = temperature
        self.llm = ChatOpenAI(api_key=api_key, temperature=self.temperature)

    def invoke(self, prompt):
        return self.llm.invoke(prompt)
