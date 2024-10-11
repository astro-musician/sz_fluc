Quelques remarques sur le code donné :

- Les différents codes se situent dans le dossier PROG. Ils appellent le package cluster_SZ qui se situe sous la même arborescence. J'ai fait un peu de ménage pour enlever certains codes tests, il doit rester les calculs de matrice de covariance, de modèle moyen, de spectre de fluctuations ainsi que différents affichages (plots des différentes cartes, cornerplot, power spectra...)
 
- Par convention, n_R500 repère la moitié du côté d'un carré centré sur le centre SZ Planck de l'amas. Cela permet de considérer uniquement un morceau centré sur l'amas.

- Les codes sont pensés pour être utilisés pour n'importe quel amas et pour n'importe quelle valeur de n_R500 (tant que le carré reste dans l'image, cad n_R500 < 10). Ces deux réglages sont généralement placés en début de code, juste après l'importation des packages.

- Les dossiers sous la même arborescence que PROG permettent de stocker les données temporaires (données Planck, matrices de covariances en .npy, cartes en .npy et en fits...). Il faut s'attendre à plusieurs Go de stockage. Je n'ai pas envoyé les matrices de transfert pour les amas XCOP car elles servent uniquement d'outil de calcul temporaire et pesaient plus de 20 Go ensemble.

- La plupart des fonctions du package créant des données stockables (matrice de transfert ou power spectra) ont une option "savefile" qui permet de choisir si la donnée doit être stockée ou non, ou bien écraser une donnée existante. Pour savefile=False, le programme considère que la donnée existe déjà et ne la calcule pas de nouveau pour gagner du temps. 

- En cas d'utilisation sur un calculateur, tous les dossiers sauf PROG doivent être placés dans le tmpdir. L'adresse du tmpdir doit alors être indiquée dans la fonction tmpdir_path du fichier pathfinder.py dans le package cluster_SZ. Le package est pensé pour répercuter l'adresse sur tous les codes utilisant le tmpdir (c'est ce que j'ai utilisé sur CALMIP).
