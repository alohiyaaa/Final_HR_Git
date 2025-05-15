# Databricks notebook source
# MAGIC %pip install PyPDF2
# MAGIC %pip install transformers
# MAGIC %pip install beautifulsoup4
# MAGIC %pip install torch
# MAGIC %pip install html2text
# MAGIC %pip install html5lib
# MAGIC %pip install tabulate

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

import os
import pandas as pd
import re
import PyPDF2
import re
from transformers import BertTokenizer
from transformers import AutoTokenizer
import math
import re
from datetime import datetime
from bs4 import BeautifulSoup
import html2text
import html5lib
import tabulate
import pandas as pd
import sys
sys.path.append("~")
import config


# COMMAND ----------


def lstrip_text(tokenizer, text):
    stripped_text = text.lstrip()
    return stripped_text

def rstrip_text(tokenizer, text):
    stripped_text = text.rstrip()
    return stripped_text

def normalize_white_spaces(tokenizer, text):
    original_token_count = len(tokenizer.encode(text))
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    return normalized_text

def remove_non_printable_characs(tokenizer, text):
    cleaned_text = re.sub(r'[^\x20-\x7E]', '', text)
    return  cleaned_text

def text_lower_case(raw_text):
    return raw_text.lower()

def preprocessing_text(tokenizer, raw_text):
    """Apply all preprocessing steps to the raw text."""
    text = lstrip_text(tokenizer, raw_text)
    text_two = rstrip_text(tokenizer, text)
    text_three = normalize_white_spaces(tokenizer, text_two)
    text_four = remove_non_printable_characs(tokenizer, text_three)
    if config.CASE_SENSITIVE_HTML:
        text_five = text_lower_case(text_four)
     
    if config.CASE_SENSITIVE_HTML:
        return text_five
    else:
        return text_four 

# COMMAND ----------

def appending_chunk(data_list, filename, embedding_window, raw_text, chunks):
    for idx, chunk in enumerate(chunks):
        data_list.append({
            'File Name': filename + "---" + str(idx),
            'Model_Max_Length': embedding_window,
            'Chunk Index': idx,
            'Text': raw_text,
            'Preprocessed_Text': chunk
        })
    return data_list

def parse_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = False
    text_maker.ignore_images = True

    text_without_tables = text_maker.handle(html_content)
    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        tables = pd.read_html(html_content, flavor='bs4')
        table_texts = [df.to_markdown(index=False) for df in tables]
        final_text = text_without_tables + "\n\n".join(table_texts)
    except ValueError:
        final_text = text_without_tables

    return final_text

def chunking(preprocess_text, tokens, num_tokens, embedding_window, tokenizer):
    chunks = []
    for i in range(0, num_tokens, embedding_window):
        chunk_tokens = tokens[i:i+embedding_window]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def process_files(folder_path, batch_size=10):
    data_list = []
    #model_name = model_path
    
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_PATH_HTML)
    list_num_tokens = []
    dict_num_tokens = {}
    list_of_chunks = []
   
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.html')]
    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i+batch_size]
        for filename in batch_filenames:
            
            file_path = os.path.join(folder_path, filename)
            raw_text = parse_html(file_path)
            raw_text_token_count = len(tokenizer.encode(raw_text))
            list_num_tokens.append(raw_text_token_count)
            preprocess_text = preprocessing_text(tokenizer, raw_text)
            tokens_returned  = tokenizer.encode(preprocess_text)
            num_tokens = len(tokens_returned) 

            if num_tokens > config.EMBEDDING_WINDOW_HTML:
                tokens = tokens_returned
                chunks = chunking(preprocess_text, tokens, num_tokens, config.EMBEDDING_WINDOW_HTML, tokenizer)
                print(chunks)
                list_of_chunks.extend(chunks)
            else:
                list_of_chunks.append(preprocess_text)
                print(preprocessing_text)

            data_list = appending_chunk(data_list, filename, config.EMBEDDING_WINDOW_HTML, raw_text, list_of_chunks)
            
    html_data = pd.DataFrame(data_list)
    return html_data

# COMMAND ----------

#folder_path = config.DBFS_FILE_PATH 
data = process_files(config.DBFS_FILE_PATH_HTML, config.BATCH_SIZE_HTML)
data.reset_index(inplace=True)
data.to_parquet(config.UPLOAD_FINAL_RESULTS_HTML)
