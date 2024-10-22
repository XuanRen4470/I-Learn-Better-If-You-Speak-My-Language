import sys
import os
import torch
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from peft import PeftModel
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import re
import torch.nn.functional as F
import os
from nltk.corpus import stopwords
import nltk




def compute_similarity_scores(model, tokenizer, ground_truths, predicted_outputs, hidden_states_layer_num, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."

    # similarity_scores = 0
    # num_pairs = len(ground_truths)
    # embedding_gt_total_list = []
    # embedding_pred_total_list = []
    # for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
    #     # Tokenize the ground truth and predicted output
    #     inputs_gt = tokenizer(ground_truth, return_tensors="pt")
    #     inputs_pred = tokenizer(predicted_output, return_tensors="pt")

    #     inputs_gt = inputs_gt.to(device)
    #     inputs_pred = inputs_pred.to(device) 

    #     # Get embeddings for ground truth and predicted output
    #     with torch.no_grad():
    #         outputs_gt = model(**inputs_gt, output_hidden_states=True)
    #         outputs_pred = model(**inputs_pred, output_hidden_states=True)
        
    #     # Hidden states is a tuple with one element per layer; the last element is the final hidden state
    #     embedding_gt = outputs_gt.hidden_states[hidden_states_layer_num].mean(dim=1)
    #     embedding_pred = outputs_pred.hidden_states[hidden_states_layer_num].mean(dim=1)
    #     # Compute cosine similarity between the ground truth and predicted output
    #     similarity = F.cosine_similarity(embedding_gt, embedding_pred)

    #     embedding_gt_total_list.append(embedding_gt)
    #     embedding_pred_total_list.append(embedding_pred)

    #     similarity_scores += similarity.item()



    

    

    # ------------
    similarity_scores = 0
    num_pairs = len(ground_truths)
    embedding_gt_total_list = []
    embedding_pred_total_list = []
    similarity_list = []

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        # Tokenize the ground truth and predicted output
        inputs_gt = tokenizer(ground_truth, return_tensors="pt")
        inputs_pred = tokenizer(predicted_output, return_tensors="pt")

        inputs_gt = inputs_gt.to(device)
        inputs_pred = inputs_pred.to(device)

        # Get embeddings for ground truth and predicted output
        with torch.no_grad():
            outputs_gt = model(**inputs_gt, output_hidden_states=True)
            outputs_pred = model(**inputs_pred, output_hidden_states=True)

        if hidden_states_layer_num == -1:
            similarities = []
            num_layers = len(outputs_gt.hidden_states)
            for layer_num in range(num_layers):
                # Get embeddings for the current layer
                embedding_gt = outputs_gt.hidden_states[layer_num].mean(dim=1)
                embedding_pred = outputs_pred.hidden_states[layer_num].mean(dim=1)

                # Compute cosine similarity for the current layer
                similarity = F.cosine_similarity(embedding_gt, embedding_pred)
                similarities.append(similarity.item())

                # Optionally collect embeddings
                embedding_gt_total_list.append(embedding_gt)
                embedding_pred_total_list.append(embedding_pred)

            # Average the similarities across all layers
            avg_similarity = sum(similarities) / len(similarities)
            similarity_scores += avg_similarity
            similarity_list.append(avg_similarity)

        else:
            # Compute embeddings and similarity for the specified layer
            embedding_gt = outputs_gt.hidden_states[hidden_states_layer_num].mean(dim=1)
            embedding_pred = outputs_pred.hidden_states[hidden_states_layer_num].mean(dim=1)
            similarity = F.cosine_similarity(embedding_gt, embedding_pred)

            embedding_gt_total_list.append(embedding_gt)
            embedding_pred_total_list.append(embedding_pred)

            similarity_scores += similarity.item()
        
            similarity_list.append(similarity.item())
        
# ---------------
    embedding_gt_total = torch.cat(embedding_gt_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]
    embedding_pred_total = torch.cat(embedding_pred_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]

    # Compute the mean embedding across all pairs
    embedding_gt_total = embedding_gt_total.mean(dim=0)  # Shape: [embedding_dim]
    embedding_pred_total = embedding_pred_total.mean(dim=0)

    avg_total_similarity = F.cosine_similarity(embedding_gt_total, embedding_pred_total, dim=0)

    similarity_scores /= num_pairs
    return similarity_scores, avg_total_similarity, similarity_list

def compute_similarity_scores_using_probabilities(model, tokenizer, ground_truths, predicted_outputs, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."
    similarity_scores = 0
    num_pairs = len(ground_truths)
    probs_gt_total_list = []
    probs_pred_total_list = []

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        # Tokenize the ground truth and predicted output
        inputs_gt = tokenizer(ground_truth, return_tensors="pt").to(device)
        inputs_pred = tokenizer(predicted_output, return_tensors="pt").to(device)

        # Get logits for ground truth and predicted output
        with torch.no_grad():
            outputs_gt = model(**inputs_gt)
            outputs_pred = model(**inputs_pred)
        
        # Get probabilities by applying softmax to the logits
        probs_gt = torch.softmax(outputs_gt.logits, dim=-1)  # Shape: [1, seq_len_gt, vocab_size]
        probs_pred = torch.softmax(outputs_pred.logits, dim=-1)  # Shape: [1, seq_len_pred, vocab_size]

        # Average probabilities over the sequence length
        probs_gt_mean = probs_gt.mean(dim=1)  # Shape: [1, vocab_size]
        probs_pred_mean = probs_pred.mean(dim=1)  # Shape: [1, vocab_size]

        # Compute cosine similarity between the probability distributions
        similarity = F.cosine_similarity(probs_gt_mean, probs_pred_mean, dim=-1)

        probs_gt_total_list.append(probs_gt_mean)
        probs_pred_total_list.append(probs_pred_mean)

        similarity_scores += similarity.item()

    # Concatenate and average the probabilities over all pairs
    probs_gt_total = torch.cat(probs_gt_total_list, dim=0)  # Shape: [num_pairs, vocab_size]
    probs_pred_total = torch.cat(probs_pred_total_list, dim=0)  # Shape: [num_pairs, vocab_size]

    # Compute the mean probability distribution across all pairs
    probs_gt_total_mean = probs_gt_total.mean(dim=0)  # Shape: [vocab_size]
    probs_pred_total_mean = probs_pred_total.mean(dim=0)  # Shape: [vocab_size]

    # Compute cosine similarity between the mean probability distributions
    avg_total_similarity = F.cosine_similarity(probs_gt_total_mean.unsqueeze(0), probs_pred_total_mean.unsqueeze(0), dim=-1)

    similarity_scores /= num_pairs
    return similarity_scores, avg_total_similarity

def get_mean_embedding_without_stopwords(model, tokenizer, input_text, hidden_states_layer_num, stop_words, device='cuda'):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids'][0]  # Shape: [seq_length]

    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[hidden_states_layer_num][0]  # Shape: [seq_length, embedding_dim]

    # Build mask for non-stop words
    mask = []
    for token in tokens:
        # Remove special tokens
        if token in tokenizer.all_special_tokens:
            mask.append(False)
            continue
        # Remove subword prefixes (e.g., '##' in BERT)
        token_stripped = token.lstrip('##').lower()
        # Check if token is a stop word
        if token_stripped in stop_words:
            mask.append(False)
        else:
            mask.append(True)
    mask = torch.tensor(mask, dtype=torch.bool, device=device)

    # Apply mask to embeddings
    embeddings = embeddings[mask, :]
    
    # Compute mean embedding
    if embeddings.size(0) > 0:
        mean_embedding = embeddings.mean(dim=0)
    else:
        # Handle cases where all tokens are stop words
        mean_embedding = torch.zeros(embeddings.size(1), device=device)
    return mean_embedding

def compute_similarity_scores_without_stopwords(model, tokenizer, ground_truths, predicted_outputs, hidden_states_layer_num, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."

    similarity_scores = 0
    num_pairs = len(ground_truths)
    embedding_gt_total_list = []
    embedding_pred_total_list = []

    # Download and set up stop words
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        # Get mean embeddings for ground truth and predicted output, excluding stop words
        embedding_gt = get_mean_embedding_without_stopwords(model, tokenizer, ground_truth, hidden_states_layer_num, stop_words, device)
        embedding_pred = get_mean_embedding_without_stopwords(model, tokenizer, predicted_output, hidden_states_layer_num, stop_words, device)

        # Compute cosine similarity
        similarity = F.cosine_similarity(embedding_gt.unsqueeze(0), embedding_pred.unsqueeze(0))
        similarity_scores += similarity.item()

        embedding_gt_total_list.append(embedding_gt)
        embedding_pred_total_list.append(embedding_pred)

    # Stack embeddings and compute the mean embeddings
    embedding_gt_total = torch.stack(embedding_gt_total_list)
    embedding_pred_total = torch.stack(embedding_pred_total_list)

    # Compute the mean embedding across all pairs
    embedding_gt_mean = embedding_gt_total.mean(dim=0)
    embedding_pred_mean = embedding_pred_total.mean(dim=0)

    avg_total_similarity = F.cosine_similarity(embedding_gt_mean.unsqueeze(0), embedding_pred_mean.unsqueeze(0))

    similarity_scores /= num_pairs
    return similarity_scores, avg_total_similarity.item()


def compute_similarity_scores_sentence_bert(model, ground_truths, predicted_outputs, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."

    similarity_scores = 0
    num_pairs = len(ground_truths)
    embedding_gt_total_list = []
    embedding_pred_total_list = []

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        # Get embeddings for ground truth and predicted output using SentenceBERT
        embedding_gt = model.encode(ground_truth, convert_to_tensor=True, device=device)
        embedding_pred = model.encode(predicted_output, convert_to_tensor=True, device=device)

        # Compute cosine similarity between the ground truth and predicted output
        similarity = F.cosine_similarity(embedding_gt, embedding_pred, dim=0)

        embedding_gt_total_list.append(embedding_gt)
        embedding_pred_total_list.append(embedding_pred)

        similarity_scores += similarity.item()

    # Stack embeddings
    embedding_gt_total = torch.stack(embedding_gt_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]
    embedding_pred_total = torch.stack(embedding_pred_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]

    # Compute the mean embedding across all pairs
    embedding_gt_mean = embedding_gt_total.mean(dim=0)  # Shape: [embedding_dim]
    embedding_pred_mean = embedding_pred_total.mean(dim=0)

    # Compute similarity between mean embeddings
    avg_total_similarity = F.cosine_similarity(embedding_gt_mean, embedding_pred_mean, dim=0)

    similarity_scores /= num_pairs
    return similarity_scores, avg_total_similarity.item()








def compute_similarity_scores_sentence_bert_individual_sentence(model, ground_truths, predicted_outputs, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."

    similarity_scores = 0
    num_pairs = 0
    embedding_gt_total_list = []
    embedding_pred_total_list = []

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        embedding_gt_total = model.encode(ground_truth, convert_to_tensor=True, device=device)
        embedding_pred_total = model.encode(predicted_output, convert_to_tensor=True, device=device)
        embedding_gt_total_list.append(embedding_gt_total)
        embedding_pred_total_list.append(embedding_pred_total)
        # Get embeddings for ground truth and predicted output using SentenceBERT

        

        ground_truth_paragraphs = ground_truth.split('\n\n')
        predicted_output_paragraphs = predicted_output.split('\n\n')

        # Step 2: Split each paragraph into sentences
        ground_truth_list = []
        predicted_output_list = []
        sentence_split_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'

        for paragraph in ground_truth_paragraphs:
            sentences = re.split(sentence_split_pattern, paragraph)
            ground_truth_list.extend(sentences)

        for paragraph in predicted_output_paragraphs:
            sentences = re.split(sentence_split_pattern, paragraph)
            predicted_output_list.extend(sentences)
        


        predicted_output_embedding_list = []
        ground_truth_embedding_list = []
        for ground_truth in ground_truth_list:
            embedding_gt = model.encode(ground_truth, convert_to_tensor=True, device=device)
            ground_truth_embedding_list.append(embedding_gt)
        
        for embedding_pred in predicted_output_list:
            embedding_pred = model.encode(predicted_output, convert_to_tensor=True, device=device)
            predicted_output_embedding_list.append(embedding_pred)
        
        similarity_score_item = 0
        for embedding_gt in ground_truth_embedding_list:
            highest_similarity = 0
            for embedding_pred in predicted_output_embedding_list:
                # Compute cosine similarity between the ground truth and predicted output
                similarity = F.cosine_similarity(embedding_gt, embedding_pred, dim=0)
                if similarity > highest_similarity:
                    highest_similarity = similarity
            similarity_score_item += highest_similarity.item()
        similarity_score_item /= len(ground_truth_embedding_list)
        similarity_scores += similarity_score_item

    # Stack embeddings
    embedding_gt_total = torch.stack(embedding_gt_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]
    embedding_pred_total = torch.stack(embedding_pred_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]

    # Compute the mean embedding across all pairs
    embedding_gt_mean = embedding_gt_total.mean(dim=0)  # Shape: [embedding_dim]
    embedding_pred_mean = embedding_pred_total.mean(dim=0)

    # Compute similarity between mean embeddings
    avg_total_similarity = F.cosine_similarity(embedding_gt_mean, embedding_pred_mean, dim=0)

    similarity_scores /= len(ground_truths)
    return similarity_scores, avg_total_similarity.item()

