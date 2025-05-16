# Databricks notebook source
# MAGIC %pip install sentence-transformers

# COMMAND ----------

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
sys.path.append("/dbfs/FileStore/Ankur")
import config

# COMMAND ----------

#embedding_model_path = "/dbfs/FileStore/huggingface/stella_en_1.5B_v5" #Stella Embedidng Hugginface Files
#Data to be embedded
#dataframe_path = "/Workspace/Users/ankur.lohiya@workday.com/PolicyBot/Parsing_Preprocessing (1)/PolicyBot_DataFrame_Docs_csv.csv"
#column = 'Preprocessed_Text'
#upload_path = "Embeddings.parque"


# COMMAND ----------

#Convert preprocessed text into embeddings
#dataframe includes the preprocessed text and embedding_model_path includes files downloaed from HuggingFace for a parituclar embedding model


def CalculateEmbeddings(dataframe, embedding_model_path):

    list_of_embeddings = []
    docs = dataframe[config.EMBEDDING_COLUMN].tolist()
    embeddings = []

    model = SentenceTransformer(config.EMBEDDING_MODEL_PATH, trust_remote_code=True).cuda()

    for i in range(len(docs)):
        doc_embeddings = model.encode(docs[i])
        embeddings.append(doc_embeddings)

    return embeddings
    

# COMMAND ----------

new_data = pd.read_csv(config.EMBEDDING_DATAFRAME_PATH)
embeddings = CalculateEmbeddings(new_data, config.EMBEDDING_MODEL_PATH)

new_data['Embeddings'] = pd.Series(embeddings)
new_data.to_parquet(config.EMBEDDING_UPLOAD_PATH)


# COMMAND ----------

new_data.shape
