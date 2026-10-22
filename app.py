from flask import Flask, render_template, request,session
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from youtube_transcript_api import YouTubeTranscriptApi

import uuid

app = Flask(__name__)
app.secret_key = "youtube_chatbot_secret"

DB_FOLDER="youtube_db"


### pydantic output model
class YoutubeAnswer(BaseModel):
    answer: str=Field(
        description="Answer based on the YouTube video transcript"
    )


#embedding model
embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

##llm model
model = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
        format="json"
    )

##PydanticOutputParser
parser = PydanticOutputParser(
    pydantic_object=YoutubeAnswer
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful YouTube video assistant.

        Instructions:
        1. Answer the user's exact question, not just the general topic.
        2. Use only the information available in the transcript context.
        3. Explain the answer in simple and clear English.
        4. If the question asks "What is...", provide the definition
           if the transcript contains it.
        5. If the context contains examples, include a relevant example.
        6. Do not give only a one-line topic description when
           the user asks for an explanation.
        7. If the answer is not available in the context, say:
           "This information was not found in the video transcript."
        8. Return ONLY valid JSON with an "answer" field.
        9. Do not return a JSON schema or Markdown code block.

        Example output:
        {{"answer": "Streaming is ..."}}

        Transcript context:
        {context}

        Formatting instructions:
        {format_instructions}
        """
    ),
    ("human", "{question}")
])



chain = prompt | model | parser

# Store retrievers for active videos 
video_retrievers = {} 
video_transcripts = {}


# Home page
@app.route("/") 
def home(): 
    return render_template( "index.html",
        transcript="",
        video_id="",
        question="",
        answer=None,
        error=None)

#### fetch transcript
@app.route("/get_transcript", methods=["GET", "POST"])
def get_transcript():

    transcript_text = ""
    error = None

    if request.method == "POST":

        video_id = request.form.get("video_id", "").strip()

        try:
            # Fetch YouTube transcript
            ytt_api = YouTubeTranscriptApi()

            transcript = ytt_api.fetch(
                video_id,
                languages=["hi", "en"]
            )

            transcript_data = transcript.to_raw_data()

            # Combine transcript text
            transcript_text = "\n".join(
                [item["text"] for item in transcript_data]
            )

             # splitting
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
            chunks = text_splitter.create_documents([transcript_text])
            
          
            ## vectorStore
            collection_name = "yt_" + uuid.uuid4().hex
            vector_store = Chroma.from_documents(
                                documents=chunks,
                                embedding=embeddings,
                                collection_name=collection_name,
                                persist_directory=DB_FOLDER
                            )
            
            ##Retriever
            retriever=vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3,"lambda_mult":0.5})

            # Save for this video
            video_retrievers[video_id] = retriever 
            video_transcripts[video_id] = transcript_text 

            session["video_id"]=video_id
            
            return render_template("index.html",
                transcript=transcript_text,
                video_id=video_id,
                question="",
                answer=None, 
                error=None)           

        except Exception as e:
            print("Answer generation error:", repr(e))
            return render_template("index.html",
                transcript="",
                video_id=video_id,
                question="",
                answer=None, 
                error = f"Transcript fetch karne mein error aaya: {str(e)}")
            
### generate answer

@app.route("/ask_question",methods=["POST"])
def ask_question():

    question= request.form.get("question", "").strip()
    video_id = session.get("video_id")

    try:
        #retriever related chunks
        retriever= video_retrievers[video_id]
        docs=retriever.invoke(question)

        context="\n\n".join(doc.page_content for doc in docs)
        print("USER QUESTION:", question)
        print("RETRIEVED CONTEXT:", context)

        if not context.strip(): 
            answer = ( "This information was not found " "in the video transcript." ) 

        else: 
            # Get formatting instructions from PydanticOutputParser
            format_instructions = parser.get_format_instructions()

            result = chain.invoke({
                "context": context,
                "question": question,
                "format_instructions": format_instructions
            })

            answer = result.answer

        return render_template(
            "index.html",
            transcript=video_transcripts[video_id],
            video_id=video_id,
            question=question,
            answer=answer,
            error=None
        )

    except Exception as e:
        return render_template( "index.html", transcript=video_transcripts.get(video_id, ""), video_id=video_id, 
        question=question, 
        answer=None, 
        error=f"Answer generate karne mein error aaya: {str(e)}" )
   

if __name__ == "__main__":
    app.run(debug=True)
