# Court Judgment Prediction and Explanation task

Given the case description of an ILDCexpert case and the corresponding predicted decision obtained with the CJP task, the goal is to predict the most important sentences that better explain the decision. To accomplish this, the occlusions method and the attention mechanism are exploited to determine which
segments of text better explain the predictions.

Two methods are applied using the two most performing models obtained by the CJP task: LegalLSGBERT+BiGRU and CaseLawBERT+BiGRU (both using the [CLS] token). Also in this case, the chunks are obtained by splitting each document of the ILDCexpert into 512 tokens with an overlap of 100.

Refers to Section 6.2 of Thesis.pdf for a detailed explanation of the methods used. 

The folders denoted as *attention_last_full* refers to the experiments in which the last chunk of each document is complete (Section 6.2 of Thesis.pdf). While the folders *experts_revised* tried to adjust some mismatches between the original ILDCexpert and the documents given to the annotators. In particular a revised version of the ILDCexpert is used. Refers to section 7.4 of Thesis.pdf for the details and to Master_Thesis/Models_whole_data/cjpe/ILDC_expert
/make_anno_csv.ipynb for the code regarding the constraction of the revised dataset.  

The file *compute_metrics.ipynb*, instead, reproduce machine vs. user explanations results. You will need the files gold_explanations_ranked.json and occ_explanations.json to generated results. You can manipulate the code according to what collections of ranks (e.g. Ranks 1 to 5) you want to compare the machine explanations with. Currently, it has been set to generate the combinations from Ranks 1 to 10.