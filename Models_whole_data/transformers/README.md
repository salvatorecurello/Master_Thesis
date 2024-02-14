# Transformers

A battery of Transformers, including both Generic models:
- RoBERTa
- BigBird
- LED

and Domain-specific models:
- LegalBERT
- CaseLawBERT
- LegalLSGBERT
- LegalLED

are employed during the experiments. The initial step involves experimentation on different sections of the ILDC documents. Following the methodology outlined in *"ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation"*, all the experiments focus on the last N tokens of the documents where N is equal to the maximum amount of tokens supported by each Transformer model. In this phase, both ILDCsingle and ILDCmulti are used.
