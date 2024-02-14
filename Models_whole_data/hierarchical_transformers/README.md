# Hierarchical Transformers Models

In this phase, all the documents contained in the ILDCmulti are split into chunks of 512 tokens with an overlapping of 100 tokens. Using Transformer models hierarchically
requires fine-tuning these models on the downstream task of classification.

Two different strategies are used:

- ILDCmulti: the last N tokens of each documents are exploited, where N is equal to the token limit of each base model (e.g., 512 for LegalBERT).
- ILDCsingle: since this dataset is smaller with respect to the ILDCmulti, a data augmentation technique is applied. In particular, to fine-tune the transformer, each document in ILDCsingle is divided into chunks of 512 tokens with 100 tokens of overlap. Then, to each chunk is assigned the same label of the whole document.

Finally, the extracted representations of the chunks (refers to folder *embedding_extraction* for details) are used as input for a sequential model that includes two layers of standard BiGRU.