from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, MeanShift


base_path = "biodcase_development_set"

modeles_classification = {
    #"Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    #"SVM": SVC(kernel='rbf', random_state=42),
    #"Réseau de Neurones (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
    "Bayes Gaussien": GaussianNB(),
    #"Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    #"Arbre Elagué": DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
}

modeles_clustering = {
    "K-Means (3 clusters)": KMeans(n_clusters=3, random_state=42, n_init='auto'),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "HDBSCAN": HDBSCAN(min_cluster_size=5),
    "Mean Shift": MeanShift(n_jobs=-1)
}
