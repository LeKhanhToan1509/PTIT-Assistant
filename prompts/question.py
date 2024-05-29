from datetime import datetime
from gg_query.tool import Tool

class Prompt:
    def __init__(self, context="", input=""):
        self.context = context
        self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.input = input

    def create_prompt(self):
        query = Tool(self.input, self.input).crawl()
        prompt_template = f"""
        Bạn tên là LKT trả lời câu hỏi bằng ngôn ngữ giống ngôn ngữ của người hỏi
        Dựa vào thông tin bạn đã biết và thông tin tôi cho bạn
        {self.context}
        và thông tin hiện có trên mạng :
        {query}
        Hiện tại là {self.date}, hãy chắt lọc thông tin ra bằng cách nếu thông tin không có trong thông tin tôi cho bạn dùng thông tin trên mạng nếu không dùng thông tin tôi cho bạn:
        Con Người :
        {self.input}
        Hãy trả lời câu hỏi 1 cách lịch sự và chính xác nhất có thể (Không cần dài dòng nhưng phải đủ ý)
        AI: 
        """
        return prompt_template



