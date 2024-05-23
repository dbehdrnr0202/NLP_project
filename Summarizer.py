
import os
from dotenv import load_dotenv
import pandas as pd
import shutil
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain, summarize
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv(verbose=True)

OPENAI_TOKEN = os.getenv('OPENAI_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_TOKEN

class Summarizer:
    def __init__(self) -> None:
        self.files_data = []
        self.llm = ChatOpenAI(temperature=0, 
                        model_name='gpt-3.5-turbo-16k')
        self.set_reduce_document_chain()
        self.set_map_chain()
        self.set_map_reduce_chain()
        
    def organize_files(categorized_df:pd.DataFrame, base_directory:str)->None:
        for category in categorized_df['category'].unique():
            category_path = os.path.join(base_directory, category)
            os.makedirs(category_path, exist_ok=True)
            
            for _, row in categorized_df[categorized_df['category'] == category].iterrows():
                shutil.move(row['path'], os.path.join(category_path, row['name']))

    def load_document(self, file_path:str):
        extension = file_path.split(".")[-1]
        if extension=='pdf':
            # PDF 파일 로드
            loader = PyPDFLoader(file_path)
            document = loader.load()
            document[0].page_content[:200]
        elif extension=='txt':
            loader = TextLoader(file_path, encoding='utf8')
            document = loader.load()
        elif extension=='docx':
            loader = Docx2txtLoader(file_path)
        return document

    def split_document(self, document):
        '''
        split_document
        '''
        # 스플리터 지정
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n\n",  # 분할기준
            chunk_size=3000,   # 사이즈
            chunk_overlap=500, # 중첩 사이즈
        )

        # 분할 실행
        split_docs = text_splitter.split_documents(document)
        # 총 분할된 도큐먼트 수
        print(f'총 분할된 도큐먼트 수: {len(split_docs)}')
        return split_docs
    
    def set_map_chain(self):
        '''
        map_chain setting
        '''
        
        # Map 단계에서 처리할 프롬프트 정의
        # 분할된 문서에 적용할 프롬프트 내용을 기입
        # 여기서 {pages} 변수에는 분할된 문서가 차례대로 대입
        map_template = """다음은 문서 중 일부 내용입니다
        {pages}
        이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
        답변:"""

        # Map 프롬프트 완성
        map_prompt = PromptTemplate.from_template(map_template)

        # Map에서 수행할 LLMChain 정의
        self.map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

    def set_reduce_document_chain(self):
        # Reduce 단계에서 처리할 프롬프트 정의
        reduce_template = """다음은 요약의 집합입니다:
        {doc_summaries}
        이것들을 바탕으로 통합된 요약을 만들어 주세요.
        답변:"""

        # Reduce 프롬프트 완성
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        text_reduce_template = """다음은 파일의 내용입니다:
        {doc_summaries}
        이를 바탕으로 요약을 만들어 주세요.
        답변:"""
        text_reduce_prompt = PromptTemplate.from_template(text_reduce_template)
        self.plain_reduce_chain = LLMChain(llm=self.llm, prompt=text_reduce_prompt)
        
        # Reduce에서 수행할 LLMChain 정의
        self.reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        # 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain,                
            document_variable_name="doc_summaries" # Reduce 프롬프트에 대입되는 변수
        )

        # Map 문서를 통합하고 순차적으로 Reduce
        self.reduce_documents_chain = ReduceDocumentsChain(
            # 호출되는 최종 체인
            combine_documents_chain=combine_documents_chain,
            # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
            collapse_documents_chain=combine_documents_chain,
            # 문서를 그룹화할 때의 토큰 최대 개수
            token_max=4000,
        )

    def set_map_reduce_chain(self):
        '''
        Map-Reduce 통합단계
        '''

        # 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합
        self.map_reduce_chain = MapReduceDocumentsChain(
            # Map 체인
            llm_chain=self.map_chain,
            # Reduce 체인
            reduce_documents_chain=self.reduce_documents_chain,
            # 문서를 넣을 llm_chain의 변수 이름(map_template 에 정의된 변수명)
            document_variable_name="pages",
            # 출력에서 매핑 단계의 결과를 반환
            return_intermediate_steps=False,
        )

    def summerize_pdf(self, file_path):
        document = self.load_pdf(file_path=file_path)
        split_docs = self.split_document(document=document)
        # Map-Reduce 체인 실행
        # 입력: 분할된 document
        result = self.map_reduce_chain.run(split_docs)
        # 요약결과 출력
        return result

    def summerize_text(self, file_path):
        document = self.load_document(file_path=file_path)
        summary = self.plain_reduce_chain.run(document)
        return summary

    
    
if __name__=="__main__":
    summarizer = Summarizer()
    summary = summarizer.summerize_text('data/feedback.txt')
    print(summary)