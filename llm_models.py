import pandas as pd
import numpy as np
from transformers import pipeline
import re
import os
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import wandb
from transformers import EarlyStoppingCallback
from sklearn.linear_model import LogisticRegression
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import EvalPrediction
from transformers import DataCollatorWithPadding

## helper functions
## data cleaning
def data_cleaning(df):
    def text_standarization(text):
        text = re.sub(r'[#@\/$*]', '', text)
        return text
    # url
    url_prefix = 'http'
    split_df = df['text'].str.split(url_prefix, expand=True)
    num_url = split_df.shape[1]
    url_cols = ['url'+str(i) for i in range(1,num_url)]
    url_title_cols = ['title'] + url_cols
    split_df.columns = url_title_cols
    if 'label' in df.columns: # works for test data
        split_df['label'] = df['label']
    for col in url_cols:
        split_df[col] = url_prefix + split_df[col]
    split_df[url_title_cols] = split_df[url_title_cols].apply(lambda col: col.str.strip()) # standarize

    # string in url
    unstructured_train_df = split_df[split_df[url_cols].apply(lambda row: row.astype(str).str.contains(' ').any(), axis=1)]
    for col in url_cols:
        unstructured_train_df_1 = unstructured_train_df[unstructured_train_df[col].str.contains(' ', na=False)]
        if unstructured_train_df_1.empty:
            continue
        else:
            unstructured_train_df_2 = unstructured_train_df_1[['title', col]]
            unstructured_train_df_2['ending_index'] = unstructured_train_df_2[col].str.find(' ')
            unstructured_train_df_2['check'] = unstructured_train_df_2.apply(
                lambda row: row[col][row['ending_index']:] if row['ending_index'] >= 0 and row['ending_index'] + 1 < len(row[col]) else '',
                axis=1
            )
            unstructured_train_df_2['title'] = unstructured_train_df_2['title'] + unstructured_train_df_2['check']
            unstructured_train_df_2[col] = unstructured_train_df_2.apply(
                lambda row: row[col][:row['ending_index']],
                axis = 1
            )
            split_df.loc[unstructured_train_df_2.index, 'title'] = unstructured_train_df_2['title']
            split_df.loc[unstructured_train_df_2.index, col] = unstructured_train_df_2[col]
    # combine url
    split_df['url'] = split_df[url_cols].apply(lambda row: row.dropna().tolist(), axis=1)
    split_df_combined = split_df.drop(columns=url_cols)
    # duplicates
    duplicate_titles = split_df_combined[split_df_combined[['title']].duplicated()]['title']
    split_df_combined_1 = split_df_combined[~split_df_combined['title'].duplicated(keep=False)]
    # label_df = pd.DataFrame(columns=['title', 'labels'])
    for title in duplicate_titles:
        df = split_df_combined[split_df_combined['title'] == title]
        urls = df['url'].values.tolist()
        flat_urls = [url for sublist in urls for url in sublist]
        unique_urls = list(set(flat_urls))
        if 'label' in df.columns:
            labels = df['label'].values.tolist()
            unique_labels = list(set(labels))
            if len(unique_labels)==1:
                unique_labels = unique_labels[0]
            new_row = pd.DataFrame({'title': [title], 'url': [unique_urls], 'label':[unique_labels]})
        else:
            new_row = pd.DataFrame({'title': [title], 'url': [unique_urls]})
        split_df_combined_1 = pd.concat([split_df_combined_1, new_row], ignore_index=True)
    split_df_combined_1 = split_df_combined_1.reset_index(drop=True)
    split_df_combined = split_df_combined_1.copy()
    # turn 'url' from list into string
    split_df_combined['url'] = split_df_combined['url'].apply(lambda row: ','.join(row))
    # standarization
    split_df_combined['title'] = split_df_combined['title'].apply(lambda row: text_standarization(row))
    return split_df_combined

## direct classification
def direct_classification(model, test_data_processed, test_inputs):
    print("Untrained model predictions:")
    print("----------------------------")
    y_pred = []
    y_test = test_data_processed['label'].values.tolist()
    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs in test_inputs['input_ids']:
            logits = model(input_ids=inputs.unsqueeze(0)).logits  
            prediction_id = torch.argmax(logits, dim=-1)
            y_pred.append(prediction_id.item())  # Store prediction
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')
    return accuracy, f1

## 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

## bert fine tuning
def fine_tune_bert(model_name, name, dataset_dict, num_train_epochs, lr_initial, weight_decay, batch_size = 8, early_stopping_patience=3, unfreeze_layer = 0, lr_scheduler_type='linear'):
    name = model_name + '-' + name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_name_1 = f"./output/{name}"
    if not os.path.exists(model_name_1):
        os.makedirs(model_name_1)
    output_dir = model_name_1
    label_list = ["Analyst Update",  "Fed and Central Banks", "Company and Product News", "Treasuries and Corporate Debt", "Dividend", "Earnings", "Energy and Oil", "Financials", "Currencies", "General News and Opinion", "Gold and Metals and Materials", "IPO", "Legal and Regulation", "M&A and Investments", "Macro", "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]
    ids = range(len(label_list))
    id2label = dict(zip(ids, label_list))
    label2id = dict(zip(label_list, ids))
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    
    # load and train
    model = (AutoModelForSequenceClassification
        .from_pretrained(model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id)
        .to(device))
    
    if unfreeze_layer > 0:
        for param in model.distilbert.transformer.layer[:-unfreeze_layer].parameters(): 
            param.requires_grad = False

    # logging_steps = len(dataset_dict['train']) // batch_size
    training_args = TrainingArguments(output_dir=output_dir,
                                    num_train_epochs=num_train_epochs,
                                    learning_rate=lr_initial,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=weight_decay,
                                    eval_strategy="epoch",
                                    disable_tqdm=False,
                                    # logging_steps=logging_steps,
                                    # push_to_hub=False,
                                    # log_level="error",
                                    # report_to=None,
                                    save_strategy="epoch",
                                    report_to=["wandb"], 
                                    load_best_model_at_end=True,
                                    metric_for_best_model="f1" ,
                                    greater_is_better=True,
                                    lr_scheduler_type=lr_scheduler_type,
                                    )
    wandb.init(project="LLM-based-text-classification", name=name)
    wandb.config.update({
        # "num_labels": len(label_list),
        "learning_rate": lr_initial,
        "batch_size": batch_size,
        "epochs": num_train_epochs,
        "weight_decay": weight_decay,
    })
    trainer = Trainer(model=model, args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=dataset_dict['train'],
                    eval_dataset=dataset_dict['valid'],
                    callbacks=callbacks
                    )
    trainer.train()
    
    # test
    test_ds = dataset_dict['test']
    accuracy_score, f1_score = evaluate_model(model, test_ds)
    wandb.log({"best_model_accu": accuracy_score, "best_model_f1": f1_score})
    # checkpoint
    metrics_df, best_accu, best_f1 = test_checkpoints(output_dir, test_ds, device, batch_size)
    wandb.log({
        'best_checkpoint_model': {'accu':best_accu, 'f1':best_f1},
        "model_params":sum(p.numel() for p in model.parameters()), 
        "unfreeze_layer": unfreeze_layer
    })
    wandb_table = wandb.Table(dataframe=metrics_df)
    wandb.log({"checkpoints_metric": wandb_table})
    
    return metrics_df

def test_checkpoints(checkpoint_dir, test_ds, device):
    best_model = None
    best_accuracy = 0
    best_f1 = 0
    model_dirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    metric_df = pd.DataFrame(columns=['model_dir', 'accuracy', 'f1'])
    for model_dir in model_dirs:
        # Load model
        model = (AutoModelForSequenceClassification
        .from_pretrained(model_dir)
        .to(device))
        accuracy, f1 = evaluate_model(model, test_ds)
        new_df = pd.DataFrame({'model_dir':[model_dir], 'accuracy':[accuracy], 'f1':[f1]})
        metric_df = pd.concat([metric_df, new_df], axis=0)
        if f1 > best_f1:
            best_accuracy = accuracy
            best_f1 = f1
            best_model = model_dir
    print(f"Best model (f1) is {best_model} with accuracy {best_accuracy} and F1 {best_f1}")
    return metric_df, best_accuracy, best_f1
      
def evaluate_model(model, test_ds, batch_size=8):
    device = torch.device("cpu")
    model = model.to(device)
    actual_labels = test_ds['label']
    input_ids = torch.tensor(test_ds['input_ids'])
    attention_mask = torch.tensor(test_ds['attention_mask'])
    
    model.eval()
    indices = list(range(len(input_ids)))
    predictions_with_indices = []
    with torch.no_grad():
        for start in range(0, len(input_ids), batch_size):
            end = min(start + batch_size, len(input_ids))
            batch_input_ids = input_ids[start:end].to(device)
            batch_attention_mask = attention_mask[start:end].to(device)
            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
            predictions = torch.argmax(logits, dim=-1)
            for idx, prediction in zip(indices[start:end], predictions.tolist()):
                predictions_with_indices.append((idx, prediction))

    predictions_with_indices.sort(key=lambda x: x[0])
    sorted_predictions = [prediction for _, prediction in predictions_with_indices]
    accuracy = accuracy_score(actual_labels, sorted_predictions)
    f1 = f1_score(actual_labels, sorted_predictions, average='weighted')

    return accuracy, f1

### gpt2 fine tuning
def compute_metrics_gpt(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": accuracy,
        "f1": f1
    }
    
def evaluate_model_gpt(model, test_ds, batch_size):
    device = torch.device("cpu")
    model = model.to(device)
    actual_labels = test_ds['label']
    input_ids = torch.tensor(test_ds['input_ids'])
    attention_mask = torch.tensor(test_ds['attention_mask'])

    model.eval()
    indices = list(range(len(input_ids)))
    # print(f'indics:\n{indices}')
    predictions_with_indices = []
    with torch.no_grad():
        for start in range(0, len(input_ids), batch_size):
            end = min(start + batch_size, len(input_ids))  #
            batch_input_ids = input_ids[start:end].to(device)
            batch_attention_mask = attention_mask[start:end].to(device)
            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
            predictions = torch.argmax(logits, dim=-1)
            for idx, prediction in zip(indices[start:end], predictions.tolist()):
                predictions_with_indices.append((idx, prediction))

    predictions_with_indices.sort(key=lambda x: x[0])
    sorted_predictions = [prediction for _, prediction in predictions_with_indices]
    accuracy = accuracy_score(actual_labels, sorted_predictions)
    f1 = f1_score(actual_labels, sorted_predictions, average='weighted')

    return accuracy, f1

def fine_tune_gpt(model_name, name, dataset_dict, num_train_epochs, lr_initial, weight_decay, batch_size = 8, early_stopping_patience=3, unfreeze_layer = 0, lr_scheduler_type='linear'):
    name = model_name + '-' + name
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_name_1 = f"./output/{name}"
    if not os.path.exists(model_name_1):
        os.makedirs(model_name_1)
    output_dir = model_name_1
    label_list = ["Analyst Update",  "Fed and Central Banks", "Company and Product News", "Treasuries and Corporate Debt", "Dividend", "Earnings", "Energy and Oil", "Financials", "Currencies", "General News and Opinion", "Gold and Metals and Materials", "IPO", "Legal and Regulation", "M&A and Investments", "Macro", "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]
    ids = range(len(label_list))
    id2label = dict(zip(ids, label_list))
    label2id = dict(zip(label_list, ids))
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    
    # load and train
    model = (AutoModelForSequenceClassification
        .from_pretrained(model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id)
        .to(device))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if unfreeze_layer > 0:
        for param in model.distilbert.transformer.layer[:-unfreeze_layer].parameters(): 
            param.requires_grad = False

    # logging_steps = len(dataset_dict['train']) // batch_size
    training_args = TrainingArguments(output_dir=output_dir,
                                    num_train_epochs=num_train_epochs,
                                    learning_rate=lr_initial,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=weight_decay,
                                    eval_strategy="epoch",
                                    disable_tqdm=False,
                                    # logging_steps=logging_steps,
                                    # push_to_hub=False,
                                    # log_level="error",
                                    # report_to=None,
                                    save_strategy="epoch",
                                    report_to=["wandb"], 
                                    load_best_model_at_end=True,
                                    metric_for_best_model="f1" ,
                                    greater_is_better=True,
                                    lr_scheduler_type=lr_scheduler_type,
                                    )
    wandb.init(project="LLM-based-text-classification", name=name)
    wandb.config.update({
        # "num_labels": len(label_list),
        "learning_rate": lr_initial,
        "batch_size": batch_size,
        "epochs": num_train_epochs,
        "weight_decay": weight_decay,
    })
    trainer = Trainer(model=model, args=training_args,
                    compute_metrics=compute_metrics_gpt,
                    train_dataset=dataset_dict['train'],
                    eval_dataset=dataset_dict['valid'],
                    callbacks=callbacks,
                    data_collator=data_collator,  
                    )
    trainer.train()
    
    # test
    test_ds = dataset_dict['test']
    accuracy_score, f1_score = evaluate_model_gpt(model, test_ds, batch_size)
    wandb.log({"best_model_accu": accuracy_score, "best_model_f1": f1_score})
    # checkpoint
    metrics_df, best_accu, best_f1 = test_checkpoints(output_dir, test_ds, device, batch_size)
    wandb.log({
        'best_checkpoint_model': {'accu':best_accu, 'f1':best_f1},
        "model_params":sum(p.numel() for p in model.parameters()), 
        "unfreeze_layer": unfreeze_layer
    })
    wandb_table = wandb.Table(dataframe=metrics_df)
    wandb.log({"checkpoints_metric": wandb_table})
    
    return metrics_df

    
## feature extractor for bert
def feature_based_prediction(dataset_dict, model, batch_size, device):
    model.eval()
    train_inputs = dataset_dict['train']
    test_inputs = dataset_dict['test']
    train_df = train_inputs.to_pandas()
    test_df = test_inputs.to_pandas()
    training_inputs = {
        'input_ids': torch.tensor(train_df['input_ids'].tolist()),
        'attention_mask': torch.tensor(train_df['attention_mask'].tolist())
    }
    test_inputs = {
        'input_ids': torch.tensor(train_df['input_ids'].tolist()),
        'attention_mask': torch.tensor(train_df['attention_mask'].tolist())
    }
    training_df_hidden = get_hidden_states(training_inputs, device, batch_size,model)
    test_df_hidden = get_hidden_states(test_inputs, device, batch_size,model) 
    x_train = training_df_hidden
    y_train = train_df['label'].values.tolist()
    x_test = test_df_hidden
    y_test = test_df['label'].values.tolist()
    # Create a Logistic Regression model (or any other classifier)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'hiddenState_logReg_accu: {accuracy:.2f}')
    print(f'hiddenState_logReg_f1: {f1:.2f}')
    
    return accuracy, f1

def get_hidden_states(inputs, device, batch_size, model):
  inputs_tensor = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
  dataloader = DataLoader(inputs_tensor, batch_size=batch_size, shuffle=False)
  all_hidden_states = []
  with torch.no_grad():
      for batch in dataloader:
          input_ids, attention_mask = batch
          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)
          inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
          last_hidden_state = model(**inputs).last_hidden_state
          batch_hidden_states = last_hidden_state[:, 0].cpu().numpy()
          all_hidden_states.append(batch_hidden_states)
  df_hidden = np.concatenate(all_hidden_states, axis=0)
  
  return df_hidden

## gpt2 prompt
def prompt_test(prompt_template, input_dict, llm, test_df, ending):
    # if isinstance(llm, GPT2LMHeadModel):
    #     llm.config.pad_token_id = llm.config.eos_token_id
    test_text = test_df['text'].values.tolist()
    actual_labels = test_df['label'].values.tolist()
    prompt_input = list(input_dict.keys())
    prompt = PromptTemplate(input_variables=prompt_input, template=prompt_template)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    pred_labels = []
    right_forms = []
    max_length = llm.pipeline.model.config.max_position_embeddings
    tokenizer = llm.pipeline.tokenizer
    for text in test_text:
        if len(tokenizer.encode(text)) > max_length:
            print(f"Warning: Input text exceeds max length for {text}")
        input_dict['text'] = text
        prediction = chain.invoke(input_dict)
        # print(f'prediction:{prediction}\n')
        predicted_label_idx = prediction.find(ending)
        if predicted_label_idx == -1:
            predicted_label_idx = prediction.rfind(ending)
            predicted_label = prediction[predicted_label_idx + len(ending):].strip()
        else:
            predicted_label = 20
        print(f'text:{text}\n')
        print(f'predict_outcome:{predicted_label}\n')
        if predicted_label.isdigit():
            predicted_label = int(predicted_label)-1
            if predicted_label>=0 and predicted_label<20:
                right_forms.append(1)
            else:
                right_forms.append(0)
        else:
          predicted_label = 20
          right_forms.append(0)
        pred_labels.append(predicted_label)
    metrics = compute_metrics(actual_labels, pred_labels)
    test_result_df = pd.DataFrame({'actual_label':[actual_labels], 'pred_label':[pred_labels], 'right_forms':[right_forms]})
    return metrics, test_result_df
