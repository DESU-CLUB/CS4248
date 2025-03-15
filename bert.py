import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from matplotlib import pyplot as plt
torch.manual_seed(0) # for reproducibility
torch.set_default_dtype(torch.float32)

# use GPU if available
if torch.cuda.is_available():
   device = torch.device("cuda")
   print('GPU')
elif torch.backends.mps.is_available(): # Apple GPU
   device = torch.device("mps")
   print('MPS')
else:
   device = torch.device("cpu")
   print("CPU")

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
text =  str(b'\xf0\x9f\x91\x89\xf0\x9f\x93\x85\xe2\x98\x80\xef\xb8\x8f\xe2\x9c\xa8\xf0\x9f\x98\x8a') # 👉📅☀️✨😊
encoded_input = tokenizer(text, return_tensors='pt').to(device)
output = model(**encoded_input)
print(output)