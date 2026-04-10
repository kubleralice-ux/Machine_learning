# Enonce
Ce mini-projet s'intéresse à une tâche de classification supervisée multi-classe de vocalisations de grandes baleines d'Antarctique. Les échantillons à traiter seront des imagettes de différentes tailles, à extraire des enregistrements audio, et il n'y a qu'une seule classe présente par image (single label). La sortie de votre modèle est donc la classe de l'évènement présent dans l'image en entrée.

Une seconde tâche plus complexe est donné en bonus : à partir des boites temps-fréquence fournis dans les fichiers d'annotation, réaliser une tâche de détection supervisée multi-classe et multi-label (plusieurs événements peuvent être présents dans un même enregistrement audio. La sortie de votre modèle est donc la classe de l'évènement ET sa localisation temporelle à partir d'un enregistrement audio en entrée.


Figure 1. Overview of the sound event detection task. Blue whale D and Z calls (BmD and BmZ) are present, as well as Fin whale 20 Hz Pulse with (Bp20p) or without (Bp20) overtone and its 40Hz downsweep (BpD).

Dans le cadre de ce projet, cette tâche est appliquée à la détection/classification de 7 types de vocalisations émises par les baleines bleues et les rorquals communs de l'Antarctique:

    Call Z (étiquette d’annotation : BmZ) : transition douce d’une fréquence unique de 27 à 16 Hz, composée de trois parties : A, B et C.

    Call A (étiquette d’annotation : BmA) : appel Z contenant uniquement la partie A.

    Call B (étiquette d’annotation : BmB) : appel Z contenant les parties A et B.

    Call D (étiquette d’annotation : BmD) : contient une composante de fréquence descendante entre 20 et 120 Hz, et peut aussi inclure des modulations supplémentaires (par exemple, une montée en fréquence au début).

    Impulsion à 20 Hz sans harmonique (étiquette d’annotation : Bp20) : contient une composante de fréquence descendante de 30 à 15 Hz.

    Impulsion à 20 Hz avec harmonique (étiquette d’annotation : Bp20plus) : impulsion de 20 Hz avec une énergie secondaire à des fréquences variables (variant entre 80 et 120 Hz).

    Descente  harmonique à 40 Hz (étiquette d’annotation : BpD) : vocalisation descendante se terminant autour de 40 Hz, généralement comprise entre 30 et 90 Hz.

Des informations plus détaillées ainsi que des exemples de spectrogrammes des vocalisations sont disponibles ici . La base de données est décrite dans [la page suivante].


# Description de la base de données 

La bade de données est  téléchargeable ici

Comme décrit dans le Tableau 1, la base de données globale intègre 11 jeux de données site-année, correspondant à des déploiements répartis tout autour de l’Antarctique, avec des périodes d’enregistrement s’étalant de 2005 à 2017. Elle contient un total de 6591 fichiers audio, représentant 1880 heures d’enregistrement, échantillonnés à 250 Hz. La base de données est composée d'un ensemble d'entrainement (train) et un ensemble de test.

L’ensemble d'entraînement est composé de tous les jeux de données site-année, à l’exception de Kerguelen 2014, Kerguelen 2015 et Casey 2017, qui ont été exclus de l'entraînement afin de constituer la base de test. Cela représente un total de 6007 fichiers audio pour l’ensemble d'entraînement, répartis sur 8 jeux de données site-année, et 587 fichiers audio pour l’ensemble de test, répartis sur 3 jeux de données site-année. 


Table 1- Description de la base de données.

Dataset	Number of audio recordings	Total duration (h)	Total events	Ratio event/duration (%)
ballenyisland2015	205	204	2222	1.4
casey2014	194	194	6866	7.3
elephantisland2013	2247	187	21223	8.6
elephantisland2014	2595	216	20964	13
greenwich2015	190	31.7	1128	6.5
kerguelen2005	200	200	2960	1.8
maudrise2014	200	83.3	2360	6.9
rosssea2014	176	176	104	5
TOTAL TRAIN	6007	1292	57827	5.1
casey2017	187	185	3263	3.3
kerguelen2014	200	200	8822	5.7
kerguelen2015	200	200	5542	3.7
TOTAL TEST	587	585	17627	5.1

Toutes les données sont annotées dans des fichiers csv fournis dans la base de données.


La structure de la base de données après téléchargement  est donnée comme suit : 

|___biodcase_development_set/

      |____train/

            |____annotations/

                  |____site_year1.csv

                  |____site_year2.csv

                  |____...

            |____audio/

                  |____site-year1/

                        |____*.wav

                  |____site-year2/

                  |____...

      |____test/

            |____annotations/

                  |____site_year1.csv

                  |____site_year2.csv

                  |____...

            |____audio/

                  |____site-year1/

                        |____*.wav

                  |____site-year2/

                  |____...

La structure du fichier d'annotation est comme suit

dataset,filename,annotation,annotator,low_frequency,high_frequency,start_datetime,end_datetime
ballenyislands2015,2015-02-04T03-00-00_000.wav,bma,nieukirk,21.9,28.4,2015-02-04T03:27:32.053000+00:00,2015-02-04T03:27:43.709000+00:00

Les coordonnées des images à extraire des spectrogrammes sont fournies par les variables : correspond elle au nom de la classe

low_frequency,high_frequency,start_datetime,end_datetime

La variable annotation correspond elle au nom de la classe.


Etapes préliminaires de préparation des données

La base de données brutes fournie au dessus n'est pas directement exploitable pour la tâche de classification d'imagettes de ce mini-projet. Les deux étapes de préparation suivantes doivent préalablement être effectuées : 

- commencer par calculer les spectrogrammes à partir des fichiers audio bruts du dataset (beaucoup de ressources disponibles pour cela, par ex ce notebook et cette librairie)

- extraire les imagettes (cad morceaux de spectrogrammes centrés sur chaque vocalisation) de ces spectrogrammes à partir des fichiers d'annotation (là aussi beaucoup de ressources comme par exemple ce tutoriel avec OpenCV)

Vous pouvez aussi vous appuyer sur ce code ici.

A noter que les imagettes auront des tailles différentes, et qu'il faudra donc mettre en place des méthodes spécifiques à ce problème. Ces méthodes peuvent aller du simple redimensionnement des imagettes pour les ramener à une taille unique, à des méthodes statistiques plus complexes comme le Pyramid Match Kernel (exemple de référence).
