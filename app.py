import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv

import numpy as np
from sklearn.cluster import KMeans

import whisper
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFium2Loader, UnstructuredPDFLoader, TextLoader, UnstructuredFileLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback



def load_files(file, loader):
    if not os.path.exists(os.path.join(os.getcwd(), "temp")):
        os.makedirs(os.path.join(os.getcwd(), "temp"))
    temp_dir = tempfile.TemporaryDirectory(dir=os.path.join(os.getcwd(), "temp"))
    print(f"Using {loader} to load {file.name}")
    temp_path = os.path.join(temp_dir.name, file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())
        print(f"Saved {file.name} to {temp_path}")
    if loader == "PyPDFium2Loader":
        # print(f"Loading {file.name} with PyPDFium2Loader")
        loader = PyPDFium2Loader(temp_path)
        pages = loader.load()
        pages = [page for page in pages if page.page_content]
        pages = [Document(page_content="\n\n".join([page.page_content for page in pages]), metadata=pages[0].metadata)]
        return pages[0]
    elif loader == "UnstructuredPDFLoader":
        # print(f"Loading {file.name} with UnstructuredPDFLoader")
        loader = UnstructuredPDFLoader(temp_path)
        pages = loader.load()
        # print(f"Loaded {len(pages)} pages")
        return pages[0]
    elif loader == "TextLoader":
        loader = TextLoader(temp_path)
        pages = loader.load()
        return pages[0]
    elif loader == "UnstructuredFileLoader":
        loader = UnstructuredFileLoader(temp_path)
        pages = loader.load()
        return pages[0]
    elif loader == "Docx2txtLoader":
        loader = Docx2txtLoader(temp_path)
        pages = loader.load()
        return pages[0]
    elif loader == "UnstructuredWordDocumentLoader":
        loader = UnstructuredWordDocumentLoader(temp_path)
        pages = loader.load()
        return pages[0]
    elif loader == "WhisperLoader":
        model = whisper.load_model("base.en")
        transcript = model.transcribe(temp_path)
        return Document(page_content=transcript["text"], metadata={"title": file.name})
    else:
        raise NotImplementedError

def chuk_files(doc, chunk_size, chunk_overlap):
    print(f"Using RecursiveCharacterTextSplitter with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(separators=["\n", "\r\n", "\r", "\t", "\n\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(doc)
    print(f"Split {len(doc)} pages into {len(chunks)} chunks")
    return chunks

def embed(docs, embeddings):
    print(f"Using {embeddings}")
    if embeddings == "HuggingFaceInstructEmbeddings":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    elif embeddings == "OpenAIEmbeddings":
        embeddings = OpenAIEmbeddings()
    else:
        raise NotImplementedError
    # vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    # print(f"Embedded {len(vectors)} chunks")
    with get_openai_callback() as cb:
        db = Chroma.from_documents(docs, embeddings)
        print(f"Embedding Callback: \n{cb}")
    return db

def load_llm(llm_name):
    if llm_name == "gpt-3.5-turbo-1106":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    elif llm_name == "gpt-4-1106-preview":
        llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    else:
        raise NotImplementedError
    return llm

def generate_quiz(llm, doc, n_questions=3):
    template = """
    Using the content provided below from a PDF delimited by triple backquotes, 
    please generate a multiple-choice test consisting of {n_questions} questions to aid students in their review. 
    Each question should be related to the key concepts, facts, or information in the text, and following the specific format below:
    Begin each question with the word "Question" and a number, followed by a colon.
    Then, provide the options for the question beginning with the word "Options" and a colon.
    Each option should begin with a letter (A, B, C, or D) followed by a period and a space.
    Ensure that for each question, there is one correct answer and three plausible but incorrect options. 
    End each question with the word "Answer" and a colon, followed by the correct answer.

    ```{text}```
    """

    # template = template.format(n_questions=n_questions)
    template = re.sub(r"{n_questions}", str(n_questions), template)
    # print(f"Using template for quiz generation: \n{template}")

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    quiz_chain = load_summarize_chain(llm, 
                                      chain_type="stuff", 
                                      prompt=prompt, 
                                      verbose=False)
 
    print(f"This doc has {llm.get_num_tokens(doc.page_content)} tokens.")
    print(f"This doc has {len(doc.page_content)} characters. {len(doc.page_content) / llm.get_num_tokens(doc.page_content)} characters per token.")
    with get_openai_callback() as cb:
        quiz = quiz_chain.invoke([doc])
        # quiz = quiz_chain.invoke()
        print(f"Generating Quiz Callback: \n{cb}")

    # print(f"Quiz: \n{quiz['output_text']}")
    questions_list = [{"question": re.split(r"\nOptions:", q)[0], 
                        "options": re.split(r"\n([A, B, C, D])", re.split(r"\nAnswer: ", re.split(r'Options:', q)[1])[0]), 
                        "answer": re.split(r"\nAnswer:", q)[1]} 
                        for q in re.split(r'Question \d+: ', quiz['output_text']) if q != ""]
    questions_list = [{"question": q["question"], "options": ["".join(x).strip() for x in zip(q["options"][1::2], q["options"][2::2])], "answer": q["answer"].strip()} for q in questions_list]

    return questions_list


def generate_quiz_map_reduce(llm, docs, n_questions=3):
    map_prompt = """
    Summarize the following text delimited by triple backquotes.
    The summary should contain the key points of the text and help students revise.
    ```{text}```
    """

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    Using the summary provided below from a PDF delimited by triple backquotes,
    please generate a multiple-choice test consisting of {n_questions} questions to aid students in their review. 
    Each question should be related to the key concepts, facts, or information in the text, and following the specific format below:
    Begin each question with the word "Question" and a number, followed by a colon.
    Then, provide the options for the question beginning with the word "Options" and a colon.
    Each option should begin with a letter (A, B, C, or D) followed by a period and a space.
    Ensure that for each question, there is one correct answer and three plausible but incorrect options. 
    End each question with the word "Answer" and a colon, followed by the correct answer.
        
    ```{text}```
    """
    combine_prompt = re.sub(r"{n_questions}", str(n_questions), combine_prompt)
    # print(combine_prompt)
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    quiz_chain = load_summarize_chain(llm,
                                    chain_type="map_reduce",
                                    map_prompt=map_prompt_template,
                                    combine_prompt=combine_prompt_template,
                                    verbose=False)
    
    with get_openai_callback() as cb:
        quiz = quiz_chain.invoke(docs)
        print(f"Generate Quiz using map reduce Callback: \n{cb}")

    # remove "Question 1: ", "Question 2: ", "question 1", etc. from the beginning of the questions
    questions_list = [{"question": re.sub(r"question? ?\d+?\n", "", re.sub(r"Question? ?\d+?\n", "", re.split(r"\nOptions:", q)[0])),
                        "options": re.split(r"\n([A, B, C, D])", re.split(r"\nAnswer: ", re.split(r'Options:', q)[1])[0]),
                        "answer": re.split(r"\nAnswer:", q)[1]}
                        for q in re.split(r'Question \d+: ', quiz['output_text']) if q != ""]
    questions_list = [{"question": q["question"], "options": ["".join(x).strip() for x in zip(q["options"][1::2], q["options"][2::2])], "answer": q["answer"].strip()} for q in questions_list]

    return questions_list

def generate_quiz_clustering(llm, vectorstore, n_questions=3):
    vectors = vectorstore.get(include=["embeddings"])["embeddings"]
    documents = vectorstore.get()["documents"]
    print(f"Using clustering to generate quiz with {len(vectors)} vectors to generate {n_questions} questions")
    num_clusters = n_questions

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    # labels_dict = {index:label for index, label in enumerate(kmeans.labels_)}
    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        
        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)
        
        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    # Print the indices of the closest embeddings to the centroids
    print(f"closest_indices: {closest_indices}")
    # print(f"labels_dict: {labels_dict}")
    # print(f"labels: {[labels_dict[index] for index in closest_indices]}")
    selected_indices = sorted(closest_indices)
    selected_docs = [Document(page_content=documents[index]) for index in selected_indices]
    template = """
    Using the content provided below from a PDF delimited by triple backquotes, 
    please generate a multiple-choice test consisting of {n_questions} questions to aid students in their review. 
    Each question should be related to the key concepts, facts, or information in the text, and following the specific format below:
    Begin each question with the word "Question" and a number, followed by a colon.
    Then, provide the options for the question beginning with the word "Options" and a colon.
    Ensure that for each question, there is one correct answer and three plausible but incorrect options. 
    End each question with the word "Answer" and a colon, followed by the correct answer.

    ```{text}```
    """
    # print(f"lem template: {len(template)}")
    template = re.sub(r"{n_questions}", str(n_questions), template)

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    quiz_chain = load_summarize_chain(llm, 
                                        chain_type="stuff", 
                                        prompt=prompt, 
                                        verbose=False)
    print(f"Processing document ...")
    print(f"This doc has {llm.get_num_tokens(selected_docs[0].page_content)} tokens.")
    print(f"This doc has {len(selected_docs[0].page_content)} characters. {len(selected_docs[0].page_content) / llm.get_num_tokens(selected_docs[0].page_content)} characters per token.")
    with get_openai_callback() as cb:
        quiz = quiz_chain.invoke(selected_docs)
        print(f"Generate Quiz using clustering Callback: \n{cb}")
    
    print(f"Generated quiz: {quiz['output_text']}")
    # remove "Question 1: ", "Question 2: ", "question 1", etc. from the beginning of the questions
    questions_list = [{"question": re.sub(r"question? ?\d+?\n", "", re.sub(r"Question? ?\d+?\n", "", re.split(r"\nOptions:", q)[0])),
                        "options": re.split(r"\n([A, B, C, D, a, b, c, d])", re.split(r"\nAnswer: ", re.split(r'Options:', q)[1])[0]),
                        "answer": re.split(r"\nAnswer:", q)[1]}
                        for q in re.split(r'Question \d+: ', quiz['output_text']) if q != ""]
    questions_list = [{"question": q["question"], "options": ["".join(x).strip() for x in zip(q["options"][1::2], q["options"][2::2])], "answer": q["answer"].strip()} for q in questions_list]

    print(f"Generated quiz: {questions_list}")
    return questions_list

def generate_feedback(llm, doc, question, user_answer):
    template = """
    Using the content provided below from a PDF delimited by triple backquotes,
    please provide feedback on the user's answer to the question below.
    
    Question: {question}
    Options: {options}
    Correct answer: {answer}
    User's answer: {user_answer}

    The feedback should be related to the key concepts, facts, or information in the text.

    ```{text}```

    Feedback:
    """

    template = re.sub(r"{question}", question["question"], template)
    template = re.sub(r"{options}", "; ".join(question["options"]), template)
    template = re.sub(r"{answer}", question["answer"], template)
    template = re.sub(r"{user_answer}", user_answer, template)
    # print(f"Using template for feedback: \n{template}")

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    feedback_chain = load_summarize_chain(llm,
                                            chain_type="stuff",
                                            prompt=prompt,
                                            verbose=False)
    with get_openai_callback() as cb:
        feedback = feedback_chain.invoke([doc])
        print(f"Generating Feedback Callback: \n{cb}")

    # delete "Feedback:", "Feedback: ", "Feedback: \n" etc. from the beginning of the feedback
    feedback = re.sub(r"Feedback:? ?\n?", "", feedback['output_text'])
    print(f"Feedback: \n{feedback}")
    return feedback


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_feedback_retrieval(llm, vectorstore, question, user_answer): 
    print(f"Generating feedback using retrieval for question: {question['question']}, user_answer: {user_answer}")


    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    template = """
    Using the content provided below from a PDF delimited by triple backquotes,
    please provide feedback on the user's answer to the question below.
    
    Question: {question}
    Options: {options}
    Correct answer: {answer}
    User's answer: {user_answer}

    The feedback should be related to the key concepts, facts, or information in the text.

    ```{context}```

    Feedback:
    """
    template = re.sub(r"{question}", question["question"], template)
    template = re.sub(r"{options}", "; ".join(question["options"]), template)
    template = re.sub(r"{answer}", question["answer"], template)
    # template = re.sub(r"{user_answer}", user_answer, template)
    # print(f"Using template for feedback: \n{template}")
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "user_answer": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    with get_openai_callback() as cb:
        feedback = rag_chain.invoke(user_answer)
        print(f"Generating Feedback using retrieval Callback: \n{cb}")

    # delete "Feedback:", "Feedback: ", "Feedback: \n" etc. from the beginning of the feedback
    feedback = re.sub(r"Feedback:? ?\n?", "", feedback)
    print(f"Feedback: \n{feedback}")
    return feedback


def main():
    load_dotenv()
    docs = []

    st.title("LLM Review Assistant")
    st.write("A tool to help you review your documents")

    #  Initialize parameters in session state
    if 'filename' not in st.session_state:
        st.session_state.filename = None
    if 'doc' not in st.session_state:
        st.session_state.doc = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = "OpenAIEmbeddings"
    if 'llm_name' not in st.session_state:
        st.session_state.llm_name = None
    if 'n_questions' not in st.session_state:
        st.session_state.n_questions = 3
    if 'generate_method' not in st.session_state:
        st.session_state.generate_method = "MapReduce"
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'large_file' not in st.session_state:
        st.session_state.large_file = False
    if 'doc_limit' not in st.session_state:
        st.session_state.doc_limit = 63_000
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'quiz' not in st.session_state:
        st.session_state.quiz = []
    if 'user_answer' not in st.session_state:
        st.session_state.user_answer = {}
    if 'user_answer_old' not in st.session_state:
        st.session_state.user_answer_old = {}
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}

    with st.sidebar:
        st.header("Settings")
        st.subheader("Process your Documents")
        # st.subheader("Your Documents")
        uploaded_file = st.file_uploader("Upload your documents", accept_multiple_files=False, type=["pdf", "txt", "docx", "mp3", "wav", "mp4"])
        if uploaded_file is None:
            st.warning("No files uploaded")
        elif uploaded_file.name.endswith(".pdf"):
            loader = st.selectbox("Choose PDF Loader", ["PyPDFium2Loader", "UnstructuredPDFLoader"])
        elif uploaded_file.name.endswith(".txt"):
            loader = st.selectbox("Choose Text Loader", ["TextLoader", "UnstructuredFileLoader"])
        elif uploaded_file.name.endswith(".docx"):
            loader = st.selectbox("Choose Word Document Loader", ["Docx2txtLoader", "UnstructuredWordDocumentLoader"])
        elif uploaded_file.name.endswith(".mp3") or uploaded_file.name.endswith(".wav") or uploaded_file.name.endswith(".mp4"):
            loader = "WhisperLoader"
        else:
            st.warning("Unsupported file type")

        if uploaded_file and uploaded_file.name != st.session_state.filename:
            with st.spinner("Loading files..."):
                st.session_state.filename = uploaded_file.name
                print(f"Uploaded files: {uploaded_file.name}")
                st.session_state.doc = load_files(uploaded_file, loader)
                print(f"Loaded {uploaded_file.name} with {loader}")

                if st.session_state.doc:
                    st.success("Files loaded!")
                else:
                    st.warning("No files uploaded")
            if len(st.session_state.doc.page_content) > 63_000:
                st.session_state.large_file = True

        if st.session_state.large_file and st.session_state.doc:
            st.warning(f"Document has {len(st.session_state.doc.page_content)} characters which will exceed GPT limits. Please split the document into smaller chunks.")
            # print(st.session_state.doc)
            print(f"The total number of characters in {uploaded_file.name} is {len(st.session_state.doc.page_content)}")
            # st.session_state.text_splitter = st.selectbox("Choose Text Splitter", ["RecursiveCharacterTextSplitter"])
            chunk_size = st.slider("Choose Chunk Size", min_value=5000, max_value=min(len(st.session_state.doc.page_content)//2, 50000), value=20000, step=1000)
            chunk_overlap = st.slider("Choose Chunk Overlap", min_value=0, max_value=chunk_size//2, value=1000, step=500)
            if st.button("Split"):
                st.spinner("Splitting files...")
                st.session_state.chunks = chuk_files([st.session_state.doc], chunk_size, chunk_overlap)
                
                st.success("Done!")
            st.session_state.generate_method = st.selectbox("Choose Quiz Generation Method", ["MapReduce", "Clustering", "Stuff"])
        
        st.session_state.embeddings = st.selectbox("Choose Embeddings model", ["OpenAIEmbeddings", "HuggingFaceInstructEmbeddings"])
        st.session_state.llm_name = st.selectbox("LLM", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
        st.session_state.n_questions = st.slider("Number of questions", min_value=1, max_value=8, value=3, step=1)
        if st.session_state.llm_name == "gpt-4-1106-preview":
            st.session_state.doc_limit = 511_000
        elif st.session_state.llm_name == "gpt-3.5-turbo-1106":
            st.session_state.doc_limit = 63_000
        

        if (st.session_state.large_file and st.session_state.chunks) or (not st.session_state.large_file and st.session_state.doc):
            if st.button("Process"):
                if uploaded_file:
                    # The embeddings will be used when the user ask questions and get feedback
                    with st.spinner("Embedding ..."):
                        if st.session_state.large_file:
                            st.session_state.db = embed(st.session_state.chunks, st.session_state.embeddings)
                        else:
                            st.session_state.db = embed([st.session_state.doc], st.session_state.embeddings)
                    # st.write(f"Embedded {len(vectors)} chunks")
                    with st.spinner("Loading LLM..."):
                        print(f"Loading {st.session_state.llm_name}")
                        st.session_state.llm = load_llm(st.session_state.llm_name)
                    with st.spinner("Generating quiz..."):
                        print(f"Generating quiz using {st.session_state.llm_name}")
                        # TODO: refine the quiz generation process
                        if st.session_state.large_file:
                            if st.session_state.generate_method == "Stuff":
                                max_docs = st.session_state.doc_limit // chunk_size
                                num_chunks = len(st.session_state.chunks)
                                stuff_chunks = [st.session_state.chunks[i:i+max_docs] for i in range(0, num_chunks, max_docs)]
                                for i, chunks in enumerate(stuff_chunks):
                                    st.session_state.quiz += generate_quiz(st.session_state.llm, chunks, st.session_state.n_questions)
                            elif st.session_state.generate_method == "MapReduce":
                                st.session_state.quiz = generate_quiz_map_reduce(st.session_state.llm, st.session_state.chunks, st.session_state.n_questions)
                            elif st.session_state.generate_method == "Clustering":
                                if st.session_state.n_questions*chunk_size >  st.session_state.doc_limit:
                                    max_docs = st.session_state.doc_limit // chunk_size
                                    # print(f"The number of clusters is too large which will exceeding the GPT 3.5 limitation of 16k. The number of clusters is set to {n_question}.")
                                    for i in range(st.session_state.n_questions // max_docs):
                                        st.session_state.quiz += generate_quiz_clustering(st.session_state.llm, st.session_state.db, max_docs)
                                    st.session_state.quiz += generate_quiz_clustering(st.session_state.llm, st.session_state.db, st.session_state.n_questions % max_docs)
                                else:
                                    # print(f"The number of clusters is set to {n_question}.")
                                    st.session_state.quiz = generate_quiz_clustering(st.session_state.llm, st.session_state.db, st.session_state.n_questions)
                                
                            else:
                                raise NotImplementedError
                        else:
                            st.session_state.quiz = generate_quiz(st.session_state.llm, st.session_state.doc, st.session_state.n_questions)
                    # st.session_state.quiz = [[{'question': 'What are some things that users can do in order to minimize the risk of a hacker getting hold of their password?', 
                    #              'options': ['A. Use the same password for all systems', 'B. Change the default password', 'C. Write their password down', 'D. Change their password frequently'], 
                    #              'answer': 'D. Change their password frequently'}, 
                    #              {'question': 'What are some things that the system can do in order to minimize the risk of attack?', 
                    #               'options': ['A. Provide the user with a default password', 'B. Reject weak passwords', 'C. Limit log-in attempts to a maximum of three', 'D. Allow the user to create a password'], 
                    #               'answer': 'B. Reject weak passwords'}, 
                    #               {'question': 'What are some measures that can be enforced by the system to minimize the risk of attack?', 
                    #                'options': ['A. Inform users of each unsuccessful log-in attempt', 'B. Insist that users choose a dictionary word as their password', 'C. Allow unlimited log-in attempts', 'D. Enforce the user to change the default password at the first log-in'], 
                    #                'answer': 'A. Inform users of each unsuccessful log-in attempt'}]]
                    st.success("Done!")
                    print(f"Generated quiz: {st.session_state.quiz}")
                else:
                    st.write("No files uploaded")
            else:
                st.write("Click the button to process your documents")
    
    st.header("Quiz")
    if not st.session_state.quiz:
        st.write("No quiz generated")
    else:      
        for i, question in enumerate(st.session_state.quiz):
            st.subheader(f"Question {i+1}")
            # print(f"Question {i+1}: {question}")
            st.session_state.user_answer[i] = st.radio(question["question"], question["options"], key=f"question_{i}", index=None)

        if st.button("Submit"):
            for i, question in enumerate(st.session_state.quiz):
                if st.session_state.user_answer[i] == question["answer"]:
                    st.session_state.feedback[i] = "Correct!"
                else:
                    # use LLM to generate feedback on wrong answers
                    # st.session_state.feedback[f"{i}_{j}"] = f"Wrong! \n\nThe correct answer is {question['answer']}. \n\nYou answered {st.session_state.user_answer[f'{i}_{j}']}"
                    if st.session_state.user_answer_old.get(i) != st.session_state.user_answer[i]:
                        if st.session_state.large_file:
                            # use RAG to generate feedback for large files
                            st.session_state.feedback[i] = generate_feedback_retrieval(st.session_state.llm, st.session_state.db, question, st.session_state.user_answer[i])
                        else:
                            st.session_state.feedback[i] = generate_feedback(st.session_state.llm, st.session_state.doc, question, st.session_state.user_answer[i])
                        st.session_state.user_answer_old[i] = st.session_state.user_answer[i]
        
        st.header("Feedback")
        if not st.session_state.feedback:
            st.write("Submit your answers to get feedback")
        else:
            for i, question in enumerate(st.session_state.quiz):
                st.subheader(f"Question {i+1}")
                st.write(st.session_state.feedback.get(i, "Submit your answer to get feedback"))

        # TODO: add chat box to ask questions wheather related to the document or not
        # TODO: add support for rate the question quality

if __name__ == "__main__":
    main()