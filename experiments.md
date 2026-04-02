 Résultats des expériences, conclusion : 
 - Un encodeur par antenne demande trop de ressources mémoire et tend à overfitter (à voir si pas autre paramètre, fusion des résultats, upscaling ???)
 - Utiliser un seul encodeur, donne les meilleure performance par rapport aux cartes RD en entrée.
 - L'entrainement avec 4 ou 8 encodeurs n'améliore pas plus les performances. 
 - La tête de détection est fine tuned, tandis que le range angle decoder est entrainé from scratch.
 - Le remplacement de la tête de détection par un réseau neuronal multicouche (MLP) s'est avéré difficile, car cette tâche relève d'un problème de régression.
- La prédiction des points est également compliquée en raison de l'incohérence des unités d'étiquettes : La distance est exprimée en mètres et l'effet Doppler est exprimé en valeur de Doppler
- L'encodage des étiquettes dans une représentation distance-Doppler et la régression directe des points dans l'espace distance-Doppler ont donné lieu à de mauvaises performances.
 - Le patch embedded comme dans T-FFTRadNet améliore uniquement les performances que sur le validation set, le test sest beaucoup moins performant. Overfit surement
 - Changement de config (p=32 car 512 patch, peut être un peu trop grand, mha=4, layer=4, dropout=0.3) Permet de réduire l'overfit pour un petit dataset
 - (A valider) apprentissage de la pyramid de feature grâce à des convolutions et des convtranspose 2D. => Pas dingue
 - (A essayer) Swin transformer avec plusieurs encoder
 - (A réessayer) Signal ADC brute, tokenization sans patch, voir si ça permet de construire la carte RA mais très peu sur -> 512x256 token = 131 072 token, mha complexité quadratique 131 072^2 = 17179869184 IMPOSS, avec patch ça marche surement mais on perd l'information spatiale importante => FourierNet
- (A essayer) Implémentation Fourier Net 
- (A essayer) Benchmark de toute la pipeline, de voir le temps que ça prend pour produire la détection 
- (A essayer) Inférence réseau avec les données du K-MD2
- Prendre données K-MD2
- Mesurer RCS


