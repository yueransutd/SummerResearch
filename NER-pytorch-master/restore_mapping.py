import torch
import pickle

def restore_mapping(file_name) :
    f=open(file_name, 'rb')  
    mappings=pickle.load(f)  
    f.close() 

    #word_to_id = mappings['word_to_id']
    tag_to_id = mappings['tag_to_id']
    #char_to_id = mappings['char_to_id']
    return  tag_to_id

file = "./models/mapping.pkl"
print(restore_mapping(file))
'''{'O': 0, 'S-LOC': 1, 'B-PER': 2, 'E-PER': 3, 'S-ORG': 4, 'S-MISC': 5, 'B-ORG': 6, 
'E-ORG': 7, 'S-PER': 8, 'I-ORG': 9, 'B-LOC': 10, 'E-LOC': 11, 'B-MISC': 12, 
'E-MISC': 13, 'I-MISC': 14, 'I-PER': 15, 'I-LOC': 16, '<START>': 17, '<STOP>': 18}'''

