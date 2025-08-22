import os
import nltk
import string
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from bisect import bisect_right
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


DEVICE = None
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using', torch.cuda.get_device_name(DEVICE), '...')
    
PAD_IDX = 0
CLS_TOKEN = '[CLS]'  # the special [CLS] token to be prepended to each sequence
SEED = 101

tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))

def tokenise_text(str_):
    """Tokenize a string of text.

    Args:
        str_: The input string of text.

    Returns:
        list(str): A list of tokens.
    """
    # for simplicity, remove non-ASCII characters
    str_ = str_.encode(encoding='ascii', errors='ignore').decode()
    return [t for t in tokeniser.tokenize(str_.lower().translate(trans_table)) if t not in stopwords]


def prepare_dataset(filename):
    """Prepare the training/validation/test dataset.

    Args:
        filename (str): The name of file from which data will be loaded.

    Returns:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set, each element of the vector is either 0 or 1.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set, each element of the vector is either 0 or 1.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set, each element of the vector is either 0 or 1.
    """
    print('Preparing train/val/test dataset ...')
    # load raw data
    df = pd.read_csv(filename)
    # shuffle the rows
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # get the train, val, test splits
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
    Xr = df["text"].tolist()
    train_end = int(train_frac*len(Xr))
    val_end = int((train_frac + val_frac)*len(Xr))
    Xr_train = Xr[0:train_end]
    Xr_val = Xr[train_end:val_end]
    Xr_test = Xr[val_end:]

    # encode sentiment labels ('pos' and 'neg')
    yr = df["label"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(yr)
    y_train = np.array(y[0:train_end])
    y_val = np.array(y[train_end:val_end])
    y_test = np.array(y[val_end:])
    return Xr_train, y_train, Xr_val, y_val, Xr_test, y_test


def build_vocab(Xt, min_freq=1):
    """Create a list of sentences, build the vocabulary and compute word frequencies from the given text data.

    Args:
        Xr (iterable(str)): A list of strings each representing a document.
        min_freq: The minimum frequency of a token that will be kept in the vocabulary.

    Returns:
        vocab (dict(str : int)): A dictionary mapping a word/token to its index.
    """
    print('Building vocabulary ...')
    counter = Counter()
    for xt in Xt:
        counter.update(xt)
    sorted_token_freq_pairs = counter.most_common()

    # find the first index where freq=min_freq-1 in sorted_token_freq_pairs using binary search/bisection
    end = bisect_right(sorted_token_freq_pairs, -min_freq, key=lambda x: -x[1])
    vocab = {token: idx+PAD_IDX+1 for (idx, (token, freq)) in enumerate(sorted_token_freq_pairs[:end])}  # PAD_IDX is reserved for padding
    vocab[CLS_TOKEN] = len(vocab) + PAD_IDX

    print(f'Vocabulary size: {len(vocab)}')
    return vocab


class MovieReviewDataset(Dataset):
    """A Dataset to be used by a data loader.
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """
    def __init__(self, X_all, y_all, cls_idx, max_seq_len):
        # X_all, y_all are the labelled examples
        # cls_idx is the index of token '[CLS]' in the vocabulary
        # max_seq_len is the maximum length of a sequence allowed
        self.X_all = X_all
        self.y_all = y_all
        self.cls_idx = cls_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        # prepend the index of the special token '[CLS]' to each sequence
        x = [self.cls_idx] + self.X_all[idx]
        # truncate a sequence if it is longer than the maximum length allowed
        if len(x) > self.max_seq_len:
            x = x[:self.max_seq_len]
        return x, self.y_all[idx]


def collate_fn(batch):
    """Merges a list of samples to form a mini-batch for model training/evaluation.
    To be used by a data loader.
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    Xb = pad_sequence([torch.tensor(x, dtype=torch.long) for (x, _) in batch], padding_value=PAD_IDX)
    yb = torch.tensor([y for (_, y) in batch], dtype=torch.float32)
    return Xb.to(DEVICE), yb.to(DEVICE)


def get_positional_encoding(emb_size, max_seq_len):
    """Compute the positional encoding.

    Args:
        emb_size (int): the dimension of positional encoding
        max_seq_len (int): the maximum allowed length of a sequence

    Returns:
        torch.tensor: positional encoding, size=(max_seq_len, emb_size)
    """
    PE = torch.zeros(max_seq_len, emb_size)
        
    for i in range(0, max_seq_len):
        for j in range(0, int(emb_size / 2)):
            PE[i, 2*j] = np.sin(i / (10000**(2*j / emb_size)))
            PE[i, 2*j + 1] = np.cos(i / (10000**(2*j / emb_size)))
    
    return PE


"""A movie review sentiment classifier using transformers."""
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=128, ffn_size=128, num_tfm_layer=2, num_head=2, p_dropout=0.2, max_seq_len=300):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX).to(DEVICE)

        # registers the positional encoding so that it is saved with the model
        # see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer(
            "positional_encoding", get_positional_encoding(emb_size, max_seq_len), persistent=False
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(emb_size, 1)  # for binary classification

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size,
                                                        nhead=num_head,
                                                        dropout=p_dropout,
                                                        dim_feedforward=ffn_size)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=num_tfm_layer)


    def forward(self, x):
        """The forward function of SentimentClassifier.
        x: a (mini-batch) of samples, size=(SEQUENCE_LENGTH, BATCH_SIZE)
        """
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        
        # Get the positional encoding portion corresponding to the input's length
        positional_encoding = self.positional_encoding[: seq_len]
        # Add an additional dimension. No need to use expand() due to broadcasting
        positional_encoding = positional_encoding.unsqueeze(1)
        
        # A mask for ignoring padded values (0s) in the input
        mask = torch.ones(batch_size, seq_len)
        mask = (x == 0)

        x = self.token_embeddings(x)
        x = positional_encoding + x
        x = self.dropout(x)
        
        x = self.encoder(x, src_key_padding_mask=mask.T)

        # [CLS] is preprended to the beginning of each sentence
        x = x[0]

        x = self.linear(x)

        return x

def eval_model(model, dataset, batch_size=64):
    """Evaluate a trained SentimentClassifier.

    Args:
        model (SentimentClassifier): a trained model
        dataset (MovieReviewDataset): a dataset of samples
        batch_size (int): the batch_size

    Returns:
        float: The accuracy of the model on the provided dataset
    """
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        preds = []
        targets = []
        for (Xb, yb) in tqdm(dataloader):
            out = model(Xb)
            preds.append(out.cpu().numpy() > 0)
            targets.append(yb.cpu().numpy())

        score = accuracy_score(np.concatenate(targets), np.concatenate(preds).astype(np.int32))
        
    model.train()
    return score

def get_training_loss(model, dataset_val, batch_size):
    """Evaluate the model on the validation set.

    Args:
        model (SentimentClassifier): a model under training
        dataset_val (MovieReviewDataset): a validation set
        batch_size (int): the batch_size

    Returns:
        float: The total loss
    """
    
    torch.cuda.empty_cache()
    model.eval()

    loss = 0
    data_loader = DataLoader(dataset_val, batch_size=batch_size, collate_fn=collate_fn)
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for (X, y) in data_loader:
            predictions = model(X).squeeze(1) # Convert shape (64x1) -> (64)
            loss += loss_fn(predictions, y)

            # Free GPU memory allocated to local variables
            del X
            del y
            del predictions
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    model.train()

    return loss

def train_model(model, dataset_train, dataset_val, batch_size=64, num_epoch=1, learning_rate=0.0001, fmodel='best_model.pth'):
    """Train a SentimentClassifier.

    Args:
        model (SentimentClassifier): a model to be trained
        dataset_train (MovieReviewDataset): a dataset of samples (training set)
        dataset_val (MovieReviewDataset): a dataset of samples (validation set)
        batch_size (int): the batch_size
        num_epoch (int): the number of training epochs
        learning_rate (float): the learning rate
        fmodel (str): name of file to save the model that achieves the best accuracy on the validation set

    Returns:
        SentimentClassifier: the trained model
    """
    model.train()
    torch.cuda.empty_cache()
    
    loss_fn = nn.BCEWithLogitsLoss()  # the binary cross entropy loss using logits
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_batch = (len(X_train) - 1) // batch_size + 1
    print(f'{"Epoch":>10} {"Batch":>15} {"Train loss (running avg.)":>25}')

    data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    total_loss = 0
    best_acc = 0
    
    for epoch in range(0, num_epoch):
        for (X, y) in tqdm(data_loader):
            optimiser.zero_grad()

            predictions = model(X).squeeze(1) # Convert shape (64x1) -> (64)
            loss = loss_fn(predictions, y)

            loss.backward()
            optimiser.step()

            # Free GPU memory allocated to local variables
            del X
            del y
            del predictions
            del loss
            torch.cuda.empty_cache()

        total_loss += get_training_loss(model, dataset_val, batch_size)
        acc_val = eval_model(clf, dataset_val)

        if acc_val > best_acc:
            torch.save(model.state_dict(), fmodel)
            best_acc = acc_val

        print(f'{epoch:>10} {batch_size:>15} {total_loss / (epoch + 1):>25}')

    return model


if __name__ == '__main__':    
    torch.manual_seed(SEED)
    fmodel = 'best_model.pth'
    data_file = os.path.join("data", "movie_reviews_labelled.csv")
    Xr_train, y_train, Xr_val, y_val, Xr_test, y_test = prepare_dataset(filename=data_file)

    get_tokenised_docs = lambda Xr: [tokenise_text(xr) for xr in tqdm(Xr)]
    get_token_indices = lambda Xt, vocab: [[vocab[token] for token in xt if token in vocab] for xt in Xt]

    Xt_train, Xt_val, Xt_test = [get_tokenised_docs(Xr) for Xr in [Xr_train, Xr_val, Xr_test]]
    vocab = build_vocab(Xt_train + Xt_val, min_freq=5)  # you may use a different min_freq
    X_train, X_val, X_test = [get_token_indices(Xt, vocab) for Xt in [Xt_train, Xt_val, Xt_test]]
    
    max_seq_len = 500
    cls_idx = vocab[CLS_TOKEN]
    dataset_train = MovieReviewDataset(X_train, y_train, cls_idx, max_seq_len)
    dataset_val = MovieReviewDataset(X_val, y_val, cls_idx, max_seq_len)
    dataset_test = MovieReviewDataset(X_test, y_test, cls_idx, max_seq_len)

    clf = SentimentClassifier(
        len(vocab),
        emb_size = 300,
        ffn_size = 512,
        num_tfm_layer = 3,
        num_head = 6,
        p_dropout = 0.5,
        max_seq_len = max_seq_len,
    ).to(DEVICE)
    
    # you may use different values for the hyper-parameters
    clf = train_model(clf, dataset_train, dataset_val, batch_size=32, num_epoch=30, learning_rate=3e-4, fmodel=fmodel)

    # print(f'Loading model from {fmodel} ...')
    clf.load_state_dict(torch.load(fmodel, map_location=torch.device(DEVICE)))
    clf = clf.to(DEVICE)
    print(clf)
    
    acc_test = eval_model(clf, dataset_test)
    print(f'Accuracy (test): {acc_test:.4f}')
    