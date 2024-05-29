import requests, json, dotenv, os
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

class Tool:
    def __init__(self, query, objective):
        self.query = query
        self.objective = objective

    def search_links(self):
        links = []
        url = "https://google.serper.dev/search"

        payload = json.dumps({
        "q": self.query,
        "gl": "vn",
        "hl": "vi"
        })
        headers = {
        'X-API-KEY': os.environ.get('SERP_API_KEY'),
        'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        item = response_json['organic']

        for i in item:
            links.append(i['link'])

        return links
    
    def crawl(self):
        links = self.search_links()
        if not links:
            return "No links found"
    
        url = links[0]
        prompt_template = """Viết một bản tóm tắt ngắn gọn khoảng 400 chữ dựa vào nôi dung:
        "{text}"
        TÓM TẮT RÚT GỌN:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = ChatOpenAI(temperature=0)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='text')
        loader = WebBaseLoader(url)

        docs = loader.load()
        data = stuff_chain.run(docs)
        return data
