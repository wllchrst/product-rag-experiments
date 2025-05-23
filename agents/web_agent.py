import requests
import traceback
import re
import numpy as np
from googlesearch import search
from bs4 import BeautifulSoup
from agents.base_agent import BaseAgent
from interfaces import ProductInformation
from googlesearch import search
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helpers import EmbedHelper
class WebAgent(BaseAgent):
    def __init__(self, config_key = None):
        super().__init__(config_key)
        self.embed_helper = EmbedHelper()
    
    def execute_task(self, data: dict) -> str:
        input = ProductInformation(**data)
        documents = self.get_feedback(keyword=input.name)

        results = self.embed_helper.get_text_similar(target=f'What is {input.name}', texts=documents)
        if results is None:
            return ""

        information = ""
        for result in results:
            information += result[0] + "\n"

        conclusion = self.extract_conclusion(information, input.name)
        return conclusion
    
    def extract_conclusion(self, information: str, product_name: str):
        prompt = f"""
I have gained information from website about a product with name {product_name}. Can you help me conclude what is the product about? Note that the information may be random and irrelevant

Information:
{information}
        """
        return self.llm.answer(prompt)
        
    def get_feedback(self, keyword: str) -> list[str]:
        documents = []
        links = self.google_search_links(keyword)
        if len(links) == 0:
            print('No links found')
            return ''

        for link in links:
            if not link or not link.startswith(('http://', 'https://')):
                print(f'Skipping invalid link: {link}')
                continue
            elif "tokopedia" in link or "shopee" in link:
                continue
            data = self.parse_url(link)

            if data is None:
                continue

            documents = documents + data

        return documents

    def google_search_links(self, question: str, number_of_result=5):
        try:
            links = []
            for link in search(question, num_results=number_of_result):
                if link == '':
                    continue
                links.append(link)

            return links
        except Exception as e:
            print(f'Error searching Google: {e}')
            traceback.print_exc()
            return []
    
    def parse_url(self, url: str) -> list[str]:
        try:
            print(f'Fetching: {url}')
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            splitted_text = self.process_page_text(soup.get_text())

            return splitted_text
        except Exception as e:
            print(f'Error parsing URL {url}: {e}')
            traceback.print_exc()
            return None
    
    def process_page_text(self, text: str) -> list[str]:
        text = re.sub(r'\n+', '', text)
        text = re.sub(r'\t+', '', text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,         # Max characters in a chunk
            chunk_overlap=10,      # Overlap between chunks
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_text(text)
        return chunks