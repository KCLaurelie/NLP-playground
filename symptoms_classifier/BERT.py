import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW
from tqdm import tqdm, trange
import pandas as pd
import io
import os
import numpy as np
import matplotlib.pyplot as plt

annots_path = '/home/ubuntu/data/ACL2020/'
# % matplotlib inline


# ## DATASETS


annots_mimic_status = annots_path + 'mimic_status_10folds.csv'
annots_mimic_temporality = annots_path + 'mimic_temporality_10folds.csv'
annots_clef_neg = annots_path + 'clef_negation_10folds.csv'
annots_clef_uncertainty = annots_path + 'clef_uncertainty_10folds.csv'
annots_attention = annots_path + 'attention_10fold.csv'

annots_file = annots_mimic_status
df = pd.read_csv(annots_file)
df.head()


# ## FUNCTIONS TO PREP DATASET / TRAIN BERT


def prep_BERT_dataset(sentences, labels=None, BERT_tokenizer='bert-base-uncased', debug=True):
    # load relevant data and add special tokens for BERT to work properly
    sentences = ["[CLS] " + query + " [SEP]" for query in sentences]
    if labels is not None:
        labels = labels.replace(-1, 0).astype('category').cat.codes.astype('long')  # convert annotations to integers
    else:
        labels = pd.Series([1] * len(sentences))
    if debug: print(sentences[0])

    # Tokenize with BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_tokenizer, do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
    if debug:
        print("Tokenize the first sentence:")
        print(len(tokenized_texts), len(input_ids), len(input_ids[0]), len(input_ids[1]), len(input_ids[2]))
        print(tokenized_texts[0], input_ids[0])

    # add paddding to input_ids
    input_ids_padded = pad_sequence([torch.tensor(i) for i in input_ids]).transpose(0, 1)
    if debug: print(input_ids_padded.size(), len(input_ids_padded))

    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids_padded:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # create dataset
    dataset = TensorDataset(input_ids_padded, torch.tensor(attention_masks), torch.tensor(labels))
    num_labels = len(labels.unique())
    return [dataset, num_labels]


# Create the DataLoaders for our training and validation sets.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
def create_BERT_dataloader(train_dataset, val_dataset, batch_size=32):
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return [train_dataloader, validation_dataloader]


# Function to calculate performance of our predictions vs labels
def nn_print_perf(preds, labels, average='weighted', debug=False):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    acc = accuracy_score(pred_flat, labels_flat)
    f1 = f1_score(pred_flat, labels_flat, average=average)
    p = precision_score(pred_flat, labels_flat, average=average)
    r = recall_score(pred_flat, labels_flat, average=average)
    if debug:
        print("PERF -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}".format(acc, f1, p, r))
    return {'f1': f1, 'acc': acc, 'p': p, 'r': r}


def run_BERT(model, train_dataloader, validation_dataloader, epochs=5, output_dir=None):
    ###################################################################################
    # BERT fine-tuning parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    # optimizer = AdamW(model.parameters(), lr=2e-5, eps = 1e-8)
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # BERT training loop
    best_f1, best_epoch = 0, 0
    for _ in trange(epochs, desc="Epoch"):
        ###################################################################################
        ## TRAINING

        # Set our model to training mode
        model.train()
        # Tracking variables
        tr_loss, tr_f1, tr_acc, tr_p, tr_r = 0, 0, 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            tmp_tr_perf = nn_print_perf(logits.detach().numpy(), b_labels.numpy(), average='weighted')
            tr_f1 += tmp_tr_perf['f1']
            tr_acc += tmp_tr_perf['acc']
            tr_p += tmp_tr_perf['p']
            tr_r += tmp_tr_perf['r']
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("TRAIN - Loss: {:.3f} - F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}"
              .format(tr_loss / nb_tr_steps, tr_f1 / nb_tr_steps, tr_acc / nb_tr_steps, tr_p / nb_tr_steps,
                      tr_r / nb_tr_steps))

        ###################################################################################
        ## VALIDATION

        # Put model in evaluation mode
        model.eval()
        # Tracking variables
        eval_loss, eval_f1, eval_acc, eval_p, eval_r = 0, 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            batch = tuple(t for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                # Update tracking variables
            tmp_eval_perf = nn_print_perf(logits.detach().numpy(), b_labels.numpy(), average='weighted')
            eval_f1 += tmp_eval_perf['f1']
            eval_acc += tmp_eval_perf['acc']
            eval_p += tmp_eval_perf['p']
            eval_r += tmp_eval_perf['r']
            nb_eval_steps += 1
        print("TEST -- F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}"
              .format(eval_f1 / nb_eval_steps, eval_acc / nb_eval_steps, eval_p / nb_eval_steps,
                      eval_r / nb_eval_steps))

        # store perf metrics and model
        if eval_f1 / nb_eval_steps >= best_f1:
            best_f1 = eval_f1 / nb_eval_steps
            best_epoch = _ + 1
            stats_to_save = {'f1': best_f1, 'acc': eval_acc / nb_eval_steps, 'p': eval_p / nb_eval_steps,
                             'r': eval_r / nb_eval_steps}
            model_to_save = model
        print('best F1 score obtained: {:.3f} at epoch {}'.format(best_f1, best_epoch))

        # # plot training performance
        # plt.figure(figsize=(15,8))
        # plt.title("Training loss")
        # plt.xlabel("Batch")
        # plt.ylabel("Loss")
        # plt.plot(train_loss_set)
        # plt.show()

    # save model with best f1
    if os.path.isdir(str(output_dir)):
        print('saving model...')
        model_to_save.save_pretrained(output_dir)
    else:
        print('model not saved, please enter valid path')

    return {'stats': stats_to_save, 'model': model_to_save}


# TO RUN K-FOLD VALIDATION
def BERT_KFOLD(sentences, labels, n_splits=10, BERT_tokenizer='bert-base-uncased', random_state=42, epochs=4,
               output_dir=None):
    dataset, num_labels = prep_BERT_dataset(sentences=sentences, labels=labels, BERT_tokenizer=BERT_tokenizer)
    pretrained_model = BertForSequenceClassification.from_pretrained(BERT_tokenizer, num_labels=num_labels)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # tracking variables
    best_f1, fold_nb = 0, 0
    stats_df = pd.DataFrame()
    # run k fold
    for train_ix, test_ix in kfold.split(labels, labels):
        fold_nb += 1
        print('####################### RUNNING FOLD:', fold_nb)
        train_dataset = torch.utils.data.Subset(dataset, train_ix)
        val_dataset = torch.utils.data.Subset(dataset, test_ix)
        print(type(train_dataset), ' train set:', len(train_dataset), ' test set:', len(val_dataset))
        train_dataloader, validation_dataloader = create_BERT_dataloader(train_dataset, val_dataset)
        res = run_BERT(pretrained_model, train_dataloader, validation_dataloader, epochs=epochs, output_dir=None)

        # store perf metrics and model
        stats_df = stats_df.append(pd.DataFrame([res['stats']]))
        if res['stats']['f1'] >= best_f1:
            best_f1 = res['stats']['f1']
            res_to_save = res

    # save model with best f1
    if os.path.isdir(str(output_dir)):
        print('saving model...')
        res_to_save['model'].save_pretrained(output_dir)
    else:
        print('model not saved, please enter valid path')

    print('best F1 score obtained across splits: {:.3f}'.format(best_f1))
    return {'stats': stats_df, 'model': res_to_save['model']}


# load pre-trained model and classify a new sentence
def load_and_run_BERT(pretrained_model_dir, sentences, BERT_tokenizer='bert-base-uncased'):
    model = BertForSequenceClassification.from_pretrained(pretrained_model_dir)
    sentences_dataset, _ = prep_BERT_dataset(sentences, labels=None, BERT_tokenizer=BERT_tokenizer)
    dataloader = DataLoader(sentences_dataset)
    b_input_ids, b_input_mask, b_labels = sentences_dataset.tensors
    with torch.no_grad():
        (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    preds = logits.detach().numpy()

    # put results in nice format
    res = pd.DataFrame()
    res['preds'] = np.argmax(preds, axis=1).flatten()
    res['sentences'] = sentences
    return res


def tests():
    BERT_tokenizer = 'bert-base-uncased'
    # BERT_tokenizer = 'emilyalsentzer/Bio_ClinicalBERT'
    # BERT_tokenizer = 'monologg/biobert_v1.0_pubmed_pmc'

    # #### RUN ALL IN 1 GO (K-FOLD)
    test = BERT_KFOLD(sentences=df.clean_text, labels=df.annotation, n_splits=10, BERT_tokenizer=BERT_tokenizer,
                      epochs=5,
                      random_state=666)
    test['model'].save_pretrained('/home/ubuntu/data/ACL2020/bert_models')

    # #### RUN ONLY 1 SIMULATION
    # prepare data
    dataset, num_labels = prep_BERT_dataset(sentences=df.clean_text, labels=df.annotation,
                                            BERT_tokenizer=BERT_tokenizer)
    # split into train/test
    test_size = 0.1
    test_len = int(len(dataset) * test_size)
    train_len = len(dataset) - test_len
    print('test set:', test_len, 'train set:', train_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    # Create the DataLoaders for our training and validation sets.
    train_dataloader, validation_dataloader = create_BERT_dataloader(train_dataset, val_dataset)
    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(BERT_tokenizer, num_labels=num_labels)
    # train and evaluate BERT
    res = run_BERT(model, train_dataloader, validation_dataloader,
                   output_dir='/home/ubuntu/data/ACL2020/bert_models/base')

    # test on new data
    sentences = df.head(5).clean_text  # put your new sentences here
    load_and_run_BERT('/home/ubuntu/data/ACL2020/bert_models/base', sentences, BERT_tokenizer='bert-base-uncased')
