# Suggestions
Notes on different ideas to explore with this challenge.

# Activity Cliff Masking
Do some exploration to identify the potential of the presence of activity cliffs in the training data. If found, implement a cliff search procedure as outlined using Structure-Activity Landscape Index (SALI) z-scores.<sup>1</sup>

$$
SALI = \frac{|A_i - A_j|}{1 - Tan_{sim}(i, j)}
$$

Fischer et al. replaced the 1 with 1.05, as stereoisomers would lead to a zero denominator. However, you can include chirality in the morgan fingerprints, so is this still a problem? I can test and see. Alas, they found that using a thresholf of SALI |z-score| > 2.5 for removing compounds lead to 95% - 98% retention of the training data and statistically improved model performance in some metrics. This is somehting I can implement, but if we want models to understand activity cliffs and why different stereoisomers have different activities, this workaround can't be something we keep sweeping under the rug. What if we asked one of these models to predict the activities of two candidate stereoisomers and one could've been a hit?

# Classical Machine-Learning
## Molecular Representations: Classical Descriptors
- Mordred descriptor set (~1800 topological, geometric, spatial etc.)
- Dragon ...
- RDKit descriptors (already using)
- Fingerprints
    - Morgan ECFP (bitsizes (256 - 2048), radius (0 - 4)), environment-based
    - Avalon fingerprint, substructure based
    - RDKit fingerprint, path-based

### Feature Selection
- autocorrelation filtering (can be implemented with `scikit-learn`)
- variance thresholds (already implemented with `scikit-learn`)

## Model Architectures
- Random Forests
- XGBoost
- LightGBM
- SVM (SVR)
- MLP ?

# Deep learning
## Molecular Representations: Embeddings
- MolFormer (molecular language model transformer pre-trained on 1.1 billion SMILES).<sup>4</sup>
- ChemBERTa (transformer-based model pre-trained on 77 million SMILES).<sup>5</sup>
- CheMeleon (MPNN pre-trained on 1 million sets of Mordred descriptors).<sup>6</sup>
- CDDD - continuous data-driven descriptors, based on an autoencoder trained on 72 million SMILES.<sup>7</sup>
- Fastprop - Deep QSAR combining mordred descriptors with deep learning. <sup>8</sup>
- Chemprop - D-MPNN

## Model Architectures
- Graph neural networks
    - MPNNs (D-MPNN)
    - GATs
- Transformer?

# Hyperparameter optimization
At some point after identifying some promising models via their 'out-of-the-box' performance, I could perform some hyperparameter optimisation via a grid method or bayesian optimisation to squeeze out some extra performance from the models. Just note that it can lead to overfitting.

# Cross-Validation
I of course need an internal way to benchmark my results to compare their performance ahead of submission. So I will choose to implement a 5 x 5 CV procedure for this.<sup>2</sup>


# References
Unfortunately, these ideas haven't been miraculously thunk up by my marvelous brain, they come from reading. Pointing myself to the references where I got these ideas from.

1. Fischer, Yaëlle, Thibaud Southiratn, Dhoha Triki, and Ruel Cedeno. "Deep Learning vs Classical Methods in Potency & ADME Prediction: Insights from a Computational Blind Challenge." (2025).
2. Ash, Jeremy R., Cas Wognum, Raquel Rodríguez-Pérez, Matteo Aldeghi, Alan C. Cheng, Djork-Arné Clevert, Ola Engkvist et al. "Practically significant method comparison protocols for machine learning in small molecule drug discovery." Journal of chemical information and modeling 65, no. 18 (2025): 9398-9411.
4. Ross, Jerret, Brian Belgodere, Vijil Chenthamarakshan, Inkit Padhi, Youssef Mroueh, and Payel Das. "Large-scale chemical language representations capture molecular structure and properties." Nature Machine Intelligence 4, no. 12 (2022): 1256-1264.
5. Ahmad, Walid, Elana Simon, Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar. "Chemberta-2: Towards chemical foundation models." arXiv preprint arXiv:2209.01712 (2022).
6. Burns, Jackson, Akshat Zalte, and William Green. "Descriptor-based Foundation Models for Molecular Property Prediction." arXiv preprint arXiv:2506.15792 (2025).
7. Winter, Robin, Floriane Montanari, Frank Noé, and Djork-Arné Clevert. "Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations." Chemical science 10, no. 6 (2019): 1692-1701.
8. Burns, Jackson W., and William H. Green. "Generalizable, fast, and accurate DeepQSPR with fastprop." Journal of Cheminformatics 17, no. 1 (2025): 73.
