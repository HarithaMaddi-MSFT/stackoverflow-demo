# +
import json
import azureml.core
from azureml.core import Workspace, Datastore, Dataset
import pandas as pd

workspace = Workspace.from_config()
print('Workspace name: ' + workspace.name, 
      'Azure region: ' + workspace.location, 
      'Subscription id: ' + workspace.subscription_id, 
      'Resource group: ' + workspace.resource_group, sep = '\n')
# -

from model import TFBertForMultiClassification
from transformers import BertTokenizer
import tensorflow as tf

datastore_config = json.loads(open('datastore.json').read())
datastore = Datastore.register_azure_blob_container(workspace=workspace, 
                                                    datastore_name=datastore_config['datastore_name'], 
                                                    container_name=datastore_config['container_name'],
                                                    account_name=datastore_config['account_name'], 
                                                    sas_token=datastore_config['sas_token'])

# If you haven't finished training the model then just download pre-made model from datastore
datastore.download('./',prefix="azure-service-classifier/model")

def encode_example(text, max_seq_length):
    # Encode inputs using tokenizer
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length
        )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    
    return input_ids, attention_mask, token_type_ids

labels = ['azure-web-app-service', 'azure-storage', 'azure-devops', 'azure-virtual-machine', 'azure-functions']
# Load model and tokenizer
loaded_model = TFBertForMultiClassification.from_pretrained('azure-service-classifier/model', num_labels=len(labels))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print("Model loaded from disk.")

# Prediction function
def predict(question):
    input_ids, attention_mask, token_type_ids = encode_example(question, 128)
    predictions = loaded_model.predict({
        'input_ids': tf.convert_to_tensor([input_ids], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor([attention_mask], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor([token_type_ids], dtype=tf.int32)
    })
    prediction = labels[predictions[0].argmax().item()]
    probability = predictions[0].max()
    result = {
        'prediction': str(labels[predictions[0].argmax().item()]),
        'probability': str(predictions[0].max())
    }
    print('Prediction: {}'.format(prediction))
    print('Probability: {}'.format(probability))


def predict_onnx(input):
    
    import json 

    # Input test sentences
    raw_data = json.dumps({
        'text': input
    })

    labels = ['azure-web-app-service', 'azure-storage', 'azure-devops', 'azure-virtual-machine', 'azure-functions']

    # Encode inputs using tokenizer
    inputs = tokenizer.encode_plus(
        json.loads(raw_data)['text'],
        add_special_tokens=True,
        max_length=max_seq_length
        )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

        # Make prediction
    convert_input = {
            sess.get_inputs()[0].name: np.array(tf.convert_to_tensor([token_type_ids], dtype=tf.int32)),
            sess.get_inputs()[1].name: np.array(tf.convert_to_tensor([input_ids], dtype=tf.int32)),
            sess.get_inputs()[2].name: np.array(tf.convert_to_tensor([attention_mask], dtype=tf.int32))
        }

    predictions = sess.run([output_name], convert_input)

    result =  {
            'prediction': str(labels[predictions[0].argmax().item()]),
            'probability': str(predictions[0].max())
        }

    print(result)


# +
import numpy as np
import onnxruntime as rt
from transformers import BertTokenizer, TFBertPreTrainedModel, TFBertMainLayer
max_seq_length = 128
labels = ['azure-web-app-service', 'azure-storage', 'azure-devops', 'azure-virtual-machine', 'azure-functions']
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

sess = rt.InferenceSession("./azure-service-classifier/model/bert_tf2.onnx")
print("ONNX Model loaded from disk.")
# -

for i in range(len(sess.get_inputs())):
    input_name = sess.get_inputs()[i].name
    print("Input name  :", input_name)
    input_shape = sess.get_inputs()[i].shape
    print("Input shape :", input_shape)
    input_type = sess.get_inputs()[i].type
    print("Input type  :", input_type)

for i in range(len(sess.get_outputs())):
    output_name = sess.get_outputs()[i].name
    print("Output name  :", output_name)  
    output_shape = sess.get_outputs()[i].shape
    print("Output shape :", output_shape)
    output_type = sess.get_outputs()[i].type
    print("Output type  :", output_type)


