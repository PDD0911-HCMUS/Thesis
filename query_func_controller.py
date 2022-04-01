import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unicodedata import name
import spacy
import pandas as pd
import requests
from spacy import displacy
# import en_core_web_sm
nlp = spacy.load('en_core_web_sm')
from spacy.tokens import Span
from spacy.matcher import Matcher
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import faiss
faiss.get_num_gpus()

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, jsonify
from flask_cors import CORS
import json

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


def extract_entities(sents):
    # chunk one
    enti_one = ""
    enti_two = ""
    
    dep_prev_token = "" # dependency tag of previous token in sentence
    
    txt_prev_token = "" # previous token in sentence
    
    prefix = ""
    modifier = ""
    
    
    
    for tokn in nlp(sents):
        # chunk two
        ## move to next token if token is punctuation
        
        if tokn.dep_ != "punct":
            #  check if token is compound word or not
            if tokn.dep_ == "compound":
                prefix = tokn.text
                # add current word to it if prevoius word is 'compound
                if dep_prev_token == "compound":
                    prefix = txt_prev_token + " "+ tokn.text
                    
            # verify if token is modifier or not
            if tokn.dep_.endswith("mod") == True:
                modifier = tokn.text
                # add it to the current word if previous word is 'compound'
                if dep_prev_token == "compound":
                    modifier = txt_prev_token + " "+ tokn.text
                    
            # chunk3
            if tokn.dep_.find("subj") == True:
                enti_one = modifier +" "+ prefix + " "+ tokn.text
                prefix = ""
                modifier = ""
                dep_prev_token = ""
                txt_prev_token = ""
                
            # chunk4 
            if tokn.dep_.find("obj") == True:
                enti_two = modifier +" "+ prefix +" "+ tokn.text
                
            # chunk 5
            # update variable
            dep_prev_token = tokn.dep_
            txt_prev_token = tokn.text
            
    return [enti_one.strip(), enti_two.strip()]

def obtain_relation(sent):
    
    doc = nlp(sent)
    
    matcher = Matcher(nlp.vocab)
    
    pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}]
    
    #matcher.add("matching_1", None, pattern)
    matcher.add("matching_1", [pattern], on_match=None)
    
    matcher = matcher(doc)
    h = len(matcher) - 1
    
    span = doc[matcher[h][1]:matcher[h][2]]
    
    return (span.text)

def knowledge_data(sent):

    pairs_of_entities = extract_entities(sent)
    relations = obtain_relation(sent)
    # subject extraction
    source = pairs_of_entities[0]

    #object extraction
    target = pairs_of_entities[1]

    edge = relations
    data_kgf = source + " " + target + " " + edge
    return data_kgf

def embbed(data_kgf):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
    embedded = []
    text = data_kgf
    marked_text = "[CLS] " + text + " [SEP]"
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []
    for token in token_embeddings:
      
      # `token` is a [12 x 768] tensor

      # Concatenate the vectors (that is, append them together) from the last 
      # four layers.
      # Each layer vector is 768 values, so `cat_vec` is length 3,072.
      cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
      
      # Use `cat_vec` to represent `token`.
      token_vecs_cat.append(cat_vec)
    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    # `hidden_states` has shape [13 x 1 x 22 x 768]

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embeddingArr = sentence_embedding.numpy()
    embedded.append(sentence_embeddingArr)
    return embedded

def faiss_index(embedded, listEmbed, kIdx = 50):
    embed_index = faiss.IndexFlatL2(768)
    res_gpu = faiss.StandardGpuResources()
    gpu_index_flat = faiss.index_cpu_to_gpu(res_gpu, 0, embed_index)
    listEmbedArr = np.asarray(listEmbed, dtype=np.float32)
    gpu_index_flat.add(listEmbedArr)
    test_fea = embedded
    # ass = []
    # ass.append(test_fea)
    # assArr = np.asarray(ass, dtype=np.float32)
    # assArr.shape
    # test_fea = assArr
    D, I = gpu_index_flat.search(test_fea, k=kIdx)
    return I

@app.route("/<filename>")
def display_media(filename):
    file_details = os.path.splitext(filename)
    file_name = file_details[0]
    file_extension = file_details[1]
    return redirect(url_for('static', filename='dataset/' + filename), code=301)

@app.route('/kg/<sent>', methods = ['GET'])
def faiss_query(sent):
    data_kgf = knowledge_data(sent)
    em = embbed(data_kgf)
    em = np.array(em)
    print(em.shape)
    dataCap = pd.read_csv('capLabel_All_KG_embedded.csv')
    dataMain = dataCap["embeded_KG"]
    listEmbed = []
    for item in dataMain:
        embedReplace = item.replace('[', '').replace(']', '')
        dataArr = np.fromstring(embedReplace, dtype=float, sep=' ')
        listEmbed.append(dataArr)
    listFile = dataCap["filename"]
    kIdx = 10
    indexQuery = faiss_index(em, listEmbed, kIdx)
    print(indexQuery)
    listImg = []
    for i in range(kIdx):
        indexIm = indexQuery[0][i]
        img = "static/dataset/" + listFile[indexIm]
        listImg.append(img)
    return jsonify(
        Data = listImg,
        Status = 200, 
        Msg = 'OK'
    ) 
        


if __name__=="__main__":
    app.run(host="192.168.0.164", port=8009)
    



