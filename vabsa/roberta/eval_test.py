from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore")

device = 'cuda'

model = RobertaForSequenceClassification.from_pretrained("checkpoint\\roberta", num_labels=4).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
test = pd.read_csv("datasets/preprocessed/roberta/prepared_test.csv")

predictions = []
labels = []
def get_dataset(df_file_csv):
    original_sentences = []
    auxiliary_sentences = []
    labels = []
    aspect_dict = {"AMBIENCE#GENERAL" : "nhận xét chung ngoại cảnh","DRINKS#PRICES": "giá đồ uống", "DRINKS#QUALITY": "chất lượng đồ uống", 
                   "DRINKS#STYLE&OPTIONS": "các lựa chọn và phong cách đồ uống" , "FOOD#PRICES": "giá đồ ăn","FOOD#QUALITY": "chất lượng đồ ăn","FOOD#STYLE&OPTIONS": "các lựa chọn và phong cách đồ ăn","LOCATION#GENERAL": "nhận xét chung vị trí",
                   "RESTAURANT#GENERAL": "nhận xét chung về nhà hàng", "RESTAURANT#MISCELLANEOUS": "khía cạnh khác của nhà hàng","RESTAURANT#PRICES": "giá nhà hàng","SERVICE#GENERAL": "nhận xét chung dịch vụ"}
    for row in range(len(df_file_csv)):
        original_sentences.append(str(df_file_csv.loc[row, "sentence"]))
        auxiliary_sentences.append(str(aspect_dict[df_file_csv.loc[row, "aspect"]]))
        labels.append((df_file_csv.loc[row, "label_id"]))
    return original_sentences, auxiliary_sentences, labels


test_original_sentences, test_auxiliary_sentences, test_labels = get_dataset(test)




pair_labels = []
pair_predicts = []
for i in tqdm(range(test.shape[0])):

    test_encodings = tokenizer(test_original_sentences[i], test_auxiliary_sentences[i], truncation=True, padding=True, return_tensors='pt').input_ids.to(device)
    logits = model(test_encodings).logits

    prediction = np.argmax(logits.cpu().detach().numpy(), axis=-1)[0]
    aspect = test.loc[i, "aspect"]
    if int(test.loc[i, "label_id"]) != 0:
       labels.append(str(aspect) + '-1')
    else:
       labels.append(str(aspect) + '-0') 
    pair_labels.append(str(aspect)+'-'+str(test.loc[i, "label_id"])) 
    if int(prediction) != 0:
       predictions.append(str(aspect) + '-1')
    else:
        predictions.append(str(aspect) + '-0')  
    pair_predicts.append(str(aspect) +'-'+str(prediction))

# target_names = ['none', 'positive','negative', 'neutral']
print(classification_report(labels, predictions, digits=4))
print(classification_report(pair_labels, pair_predicts,  digits=4))


