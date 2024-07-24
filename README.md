# Text-summarization-using-LLM
## AIM : To explore how LLMs can enhance the accuracy and efficiency of summarizing extensive text data
## ALGORITHM
Step1: Import the necessary libraries and load a pre-trained open-source LLM from a repository such as Hugging Face's Transformers.

Step2: Clean and preprocess the text data to ensure it's in a suitable format for the model.

Step3: Feed the preprocessed text into the LLM to generate summaries. This typically involves using the model’s summarization pipeline or relevant functions.

Step4: Refine and format the generated summaries as needed. This may involve detokenizing the text or adjusting the length and coherence of the summaries.

Step5: Assess the quality of the summaries using evaluation metrics or human feedback. Optionally, fine-tune the model on specific datasets to improve performance for your use case.
## PROGRAM:
```
!pip install --upgrade --quiet  langchain-openai tiktoken chromadb langchain
!pip install langchain_community
!pip install -q torch transformers accelerate bitsandbytes transformers sentence-transformers faiss-gpu
!pip install -q langchain huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "HuggingFaceH4/zephyr-7b-beta"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from langchain.llms import HuggingFacePipeline
from transformers import pipeline

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
!pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/NIHA resume.pdf")
docs = loader.load()

# Check if the issue is resolved. If not, you may need to investigate further
# to determine the acceptable content types for the server.
print(docs)
print(len(docs))
docs[0]
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm, chain_type="stuff")
chain.run(docs)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

docs = loader.load()
print(stuff_chain.run(docs))

```
## OUTPUT:
## Resume summarization: 

<img width="704" alt="image" src="https://github.com/user-attachments/assets/76245fa6-8ecb-4f57-bc13-a247d13e8cef">

## PDF meta data and content extraction : 

<img width="702" alt="image" src="https://github.com/user-attachments/assets/ccce0997-4bfd-4155-90bb-c3957ae3ae3b">

<img width="773" alt="image" src="https://github.com/user-attachments/assets/a30a9469-f025-44d2-b13c-f60c995c47a8">

## Fundamental rights textbook summarization:

<img width="794" alt="image" src="https://github.com/user-attachments/assets/4a615379-da83-4b6f-9673-6150b825add4">

## RESULT :
Thus,Large Language Models (LLMs) effectively summarize resumes by highlighting key skills and experiences, and textbooks by condensing main concepts and themes, improving readability and comprehension.

