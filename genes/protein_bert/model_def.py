import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import torch 
import torch.nn.functional as F
import torch.nn as nn

#Down model by the rostlab which already trained on the large corpus for the protein sequences~
PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
class ProteinClassifier(nn.Module):
    def __init__(self,n_classes):
        super(ProteinClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                       nn.Linear(self.bert.config.hidden_size,n_classes),
                                       nn.Tanh())
    
    def forward(self,input_ids,attention_mask):
        output = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask)
        
        return self.classifier(output.pooler_output)