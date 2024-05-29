from transformers import pipeline

class Correct:
    def __init__(self, max_length = 512):
        self.corect = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
        self.max_length = max_length
        
    def _run(self, text):
        corrected_text = self.corect(text, max_length=self.max_length)
        return corrected_text[0]['generated_text']

