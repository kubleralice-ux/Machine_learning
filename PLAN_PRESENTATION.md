# Plan de Présentation : Classification de Vocalisations de Baleines

**Durée estimée :** 13-14 minutes
**Auteurs :** WizHack, Alice, Mathis

---

## Slide 1 : Introduction et Objectifs (1 min)
**Contenu Visuel :**
*   Titre : Mission Bio-Acoustique en Antarctique.
*   Logos (ENSTA / Projet).
*   Illustration d'une baleine bleue et d'un rorqual.
*   Objectif : Classification automatique de 7 types de cris (BmA, BmZ, Bp20, etc.).

**Texte à lire :**
"Bonjour à tous. Nous allons vous présenter notre travail sur la détection et la classification des vocalisations de grandes baleines d'Antarctique. L'enjeu est écologique : utiliser l'acoustique passive pour suivre les populations de baleines bleues et de rorquals communs. Nous avons traité une base de données complexe de plus de 6000 fichiers audio, échantillonnés à 250 Hz, avec pour mission de transformer ces signaux sonores en images identifiables par des algorithmes de Machine Learning."

---

## Slide 2 : Pré-traitement - La Transformation Temps-Fréquence (2 min)
**Contenu Visuel :**
*   Schéma : Signal Audio (.wav) -> Spectrogramme.
*   Paramètres techniques : Fs=250Hz, Fenêtre de Hann, Nperseg=256, Noverlap=192.
*   Image d'un spectrogramme brut à 250Hz.

**Texte à lire :**
"La première étape, la plus critique, est le passage du domaine temporel au domaine fréquentiel. Puisque les baleines émettent à des fréquences très basses (souvent entre 15 et 100 Hz), notre fréquence d'échantillonnage est de seulement 250 Hz. Nous avons utilisé la transformée de Fourier à court terme pour générer des spectrogrammes. Nous avons opté pour une échelle logarithmique en décibels pour faire ressortir l'énergie des cris par rapport au bruit de fond de l'océan, suivi d'une normalisation min-max pour stabiliser les entrées de nos modèles."

---

## Slide 3 : Pré-traitement - Uniformisation des Imagettes (2 min)
**Contenu Visuel :**
*   Comparaison visuelle : Imagette étirée (mauvais) vs Imagette avec Padding (bon).
*   Texte : "Zéro-Padding" pour conserver l'aspect ratio.
*   Dimension finale : 64x64 pixels en niveaux de gris.

**Texte à lire :**
"Le défi majeur ici est que les cris de baleines n'ont pas tous la même durée ni la même plage de fréquences. Un simple redimensionnement 'écraserait' les caractéristiques physiques du cri. Pour éviter cela, nous avons développé un script de 'Padding'. Nous calculons la taille maximale parmi toutes les imagettes du dataset, puis nous centrons chaque cri sur un fond noir avant de ramener le tout à une résolution standard de 64x64. Cela permet de conserver la pente et la forme originale des vocalisations, ce qui est crucial pour la distinction entre un cri 'Z' et un cri 'A' par exemple."

---

## Slide 4 : Méthodes de Classification Supervisée (1 min 30)
**Contenu Visuel :**
*   Liste des modèles : SVM (RBF), Random Forest, KNN, MLP (Réseau de neurones), Gradient Boosting.
*   Outil : Scikit-Learn.
*   Configuration : Utilisation d'un fichier `config.py` pour la reproductibilité.

**Texte à lire :**
"Pour la classification, nous avons adopté une approche comparative. Plutôt que de parier sur un seul modèle, nous avons testé cinq algorithmes classiques de la littérature. Du SVM avec noyau RBF pour sa capacité à gérer les frontières non-linéaires, aux Random Forests pour leur robustesse face au sur-apprentissage. Toutes nos expériences ont été centralisées via un fichier de configuration pour garantir que chaque membre de l'équipe teste les modèles dans les mêmes conditions exactes."

---

## Slide 5 : Exploration Non-Supervisée - Clustering (1 min 30)
**Contenu Visuel :**
*   Logos : K-Means, HDBSCAN, Mean Shift.
*   Objectif : Découverte de structures sans étiquettes.
*   Méthode : PCA (Analyse en Composantes Principales) préalable pour réduire à 2 dimensions.

**Texte à lire :**
"En parallèle de la classification, nous avons mené une étude exploratoire via le clustering. L'idée était de voir si les 7 classes définies par les experts biologistes émergeaient naturellement du signal. Nous avons utilisé la PCA pour projeter nos images 64x64 (soit 4096 dimensions) sur un plan 2D, puis appliqué des algorithmes comme HDBSCAN pour détecter des densités de cris similaires et Mean Shift pour trouver les centres de gravité des clusters."

---

## Slide 6 : Résultats - Performances Globales (1 min 30)
**Contenu Visuel :**
*   Tableau comparatif des Accuracy :
    *   **SVM : 90%**
    *   Random Forest : 88%
    *   MLP : 86%
*   Mention du temps d'entraînement.

**Texte à lire :**
"Passons aux résultats. Le grand gagnant est le SVM avec une précision impressionnante de 90% sur le set de validation. Le Random Forest suit de près avec 88%. Ce qui est intéressant, c'est que même des modèles relativement simples comme le KNN obtiennent des scores honorables, ce qui valide la qualité de notre étape de pré-traitement et de padding. Si les données sont bien préparées en amont, les modèles classiques s'avèrent extrêmement efficaces."

---

## Slide 7 : Analyse Fine - Forces et Faiblesses (2 min)
**Contenu Visuel :**
*   Zoom sur la Matrice de Confusion du SVM.
*   Points Verts : BmA (97%), Bp20plus (99%).
*   Point Rouge : BpD (4%).
*   Texte : "Confusion entre BpD et Bp20".

**Texte à lire :**
"Si l'on regarde la matrice de confusion, nous avons des résultats quasi parfaits sur les classes BmA et Bp20plus. Cependant, nous avons un 'talon d'Achille' : la classe BpD. Sa précision chute à 4%. Pourquoi ? Après analyse, il s'avère que le cri BpD est très rare dans le dataset d'entraînement et ses caractéristiques fréquentielles chevauchent énormément celles du Bp20. Le modèle finit par privilégier la classe la plus fréquente par sécurité statistique. C'est un cas typique de déséquilibre de classes."

---

## Slide 8 : Analyse du Clustering (1 min 30)
**Contenu Visuel :**
*   Graphique PCA coloré par clusters (Mean Shift ou HDBSCAN).
*   Mention du score ARI (Adjusted Rand Index) : ~0.24.
*   Observation : Les clusters sont très denses et se chevauchent.

**Texte à lire :**
"Côté clustering, les résultats sont plus nuancés. Avec un score ARI de 0.24, on constate que les clusters naturels ne collent pas parfaitement aux étiquettes biologiques. Sur la visualisation PCA, on voit une grande masse centrale. Cela suggère que si les cris sont distincts pour une oreille humaine ou un classifieur supervisé, ils partagent une base acoustique commune très forte qui rend la séparation automatique 'aveugle' (sans labels) très complexe."

---

## Slide 9 : Conclusion et Perspectives (1 min 30)
**Contenu Visuel :**
*   Points clés : 90% de succès, importance du Padding.
*   Perspectives :
    *   Réseaux de Neurones Convolutifs (CNN).
    *   Augmentation de données (SMOTE) pour BpD.
    *   Détection multi-label (le bonus).

**Texte à lire :**
"Pour conclure, nous avons validé une chaîne de traitement complète, du signal audio brut à la décision automatisée. La force de notre approche réside dans l'uniformisation intelligente des images. Pour aller plus loin, l'étape logique serait de passer au Deep Learning avec des CNN. Contrairement à nos modèles actuels qui 'aplatissent' l'image, un CNN pourrait apprendre des filtres spatiaux capables de distinguer les nuances les plus subtiles des spectrogrammes, notamment pour résoudre le problème de la classe BpD. Merci de votre attention."

---

## Slide 10 : Questions / Réponses (5-7 min)
**Contenu Visuel :**
*   "Merci pour votre écoute. Avez-vous des questions ?"
*   Contact / GitHub du projet.
