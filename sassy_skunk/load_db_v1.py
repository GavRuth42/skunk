import os
import warnings
import pathlib
import re

# -------------------------------------------------
# SET YOUR OPENAI API KEY
# -------------------------------------------------
os.environ["OPENAI_API_KEY"] = "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"

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import warnings

def extract_headings_from_first_page(page_text: str) -> dict:
    """
    Extract headings from the first page using a regex.
    Example: lines starting with 'Title', 'Chapter', or 'Subchapter'
    (case-insensitive). Modify as needed for your docs.
    """
    headings = {}
    
    lines = page_text.split('\n')
    heading_regex = re.compile(r'^(Title.*|Chapter.*|Subchapter.*)$', re.IGNORECASE)
    
    count = 1
    for line in lines:
        line = line.strip()
        match = heading_regex.match(line)
        if match:
            headings[f"heading_{count}"] = match.group(1)
            count += 1

    return headings

class PersistentChromaQA:
    def __init__(
        self,
        pdf_dir1,
        pdf_dir2,
        pdf_dir3,
        pdf_dir4,
        persist_dir="chroma_db",
        chunk_size=1600,
        chunk_overlap=100,
        top_k=3,
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    ):
        self.pdf_dir1 = pdf_dir1
        self.pdf_dir2 = pdf_dir2
        self.pdf_dir3 = pdf_dir3
        self.pdf_dir4 = pdf_dir4
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature

        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="langchain"
        )

        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

        if self._is_persist_dir_available():
            print(f"Loading existing Chroma store from '{self.persist_dir}'...")
            self.vectorstore = self._load_chroma_store()
        else:
            print(f"No existing Chroma store found at '{self.persist_dir}'. Creating a new one...")
            self.vectorstore = self._create_chroma_store()

        self.qa_chain = self._build_qa_chain()

    def _is_persist_dir_available(self):
        path = pathlib.Path(self.persist_dir)
        return path.exists() and any(path.iterdir())

    def _load_chroma_store(self):
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )

    def _create_chroma_store(self):
        all_docs = self._load_all_documents()
        if not all_docs:
            raise ValueError("No documents found. Cannot create a Chroma store.")

        vectorstore = Chroma.from_documents(
            all_docs,
            self.embeddings,
            persist_directory=self.persist_dir
        )
        vectorstore.persist()
        print(f"Chroma store created and persisted at '{self.persist_dir}'.")
        return vectorstore

    def _load_all_documents(self):
        all_docs = []

        for pdf_dir in [self.pdf_dir1, self.pdf_dir2, self.pdf_dir3, self.pdf_dir4]:
            pdf_files = [f for f in pathlib.Path(pdf_dir).glob("*.pdf")]
            for pdf_file in pdf_files:
                loader = PyPDFLoader(str(pdf_file))
                try:
                    pdf_docs = loader.load()
                    title = pathlib.Path(pdf_file).stem
                    chunks = self._chunk_documents(pdf_docs, title)
                    all_docs.extend(chunks)
                    print(f"Loaded {len(pdf_docs)} pages from '{pdf_file}'.")
                except Exception as e:
                    print(f"Error loading '{pdf_file}': {e}")

        return all_docs

    # UPDATED: unify extracted headings into "heading_key"
    def _chunk_documents(self, documents, file_title):
        """
        Split the PDF pages into sub-chunks, 
        then add top-of-document headings as a single "heading_key" in each chunk's metadata.
        """
        top_headings = {}
        if len(documents) > 0:
            first_page_text = documents[0].page_content
            top_headings = extract_headings_from_first_page(first_page_text)

        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splitted_docs = text_splitter.split_documents(documents)

        for chunk in splitted_docs:
            chunk.metadata = chunk.metadata or {}
            chunk.metadata['file_title'] = file_title

            # OLD LOGIC (commented out):
            # for heading_key, heading_value in top_headings.items():
            #     chunk.metadata[heading_key] = heading_value

            # NEW LOGIC: unify all headings into a single field named "heading_key"
            all_headings = []
            for _, heading_value in top_headings.items():
                if heading_value.strip():
                    all_headings.append(heading_value.strip())

            # e.g. "Title 40 ... | Chapter I ... | Subchapter ... "
            if all_headings:
                chunk.metadata["heading_key"] = " | ".join(all_headings)

        return splitted_docs

    def _build_qa_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=self.llm_model, temperature=self.temperature),
            chain_type="stuff",
            retriever=retriever
        )
        return chain

    def query(self, user_question):
        if not user_question:
            raise ValueError("The user_question cannot be empty.")

        answer = self.qa_chain.run(user_question)
        return answer

if __name__ == "__main__":
    pdf_directory1 = "../CFR_33"
    pdf_directory2 = "../CFR_40"
    pdf_directory3 = "../CFR_49"
    pdf_directory4 = "../CFR_29"

    qa_system = PersistentChromaQA(
        pdf_dir1=pdf_directory1,
        pdf_dir2=pdf_directory2,
        pdf_dir3=pdf_directory3,
        pdf_dir4=pdf_directory4,
        persist_dir="chroma_db",
        chunk_size=1600,
        chunk_overlap=100,
        top_k=3,
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    )

    user_question = "My facility just had an oil spill of 500 gallons. What do I need to do?"
    answer = qa_system.query(user_question)

    print("\nAnswer to your question:")
    print(answer)
