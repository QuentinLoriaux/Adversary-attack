# Attaque en segmentation sémantique

### Note

Je bute complètement sur le projet. J'essaye d'implémenter le DAG et ça marche que pouic. Essentiellement parce que j'ai aucune idée de comment calculer le gradient d'une fonction de perte qui n'est pas déjà implémentée.

Pour ne pas perdre trop de temps, je mets ici des éléments pour la constitution du rapport.

## Segmentation sémantique?

Il s'agit de déterminer à quelle classe (parmi celles étudiées) appartient chaque pixel d'une image. On obtient ainsi plusieurs amas de pixels correspondant chacun à une classe.

En réalité, à chaque pixel est associée une liste de scores correspondant aux classes, et c'est la classe au plus grand score qui l'emporte.

Cela permet de connaître la nature des objets apparaissant à l'écran, mais pas forcément de les énumérer comme le ferait la détection d'objets. On le constate avec l'exemple des chiens : ils sont 3 mais on ne distingue qu'un seul groupe de pixels "chiens" par segmentation sémantique.

## Q1 : créer une attaque "untargeted" contre 1 réseau

Nous avons à notre disposition un réseau ResNet avec des poids déjà calculés, entraîné pour repérer des animaux ou objets du quotidien : chien, mouton, humain, avion, sofa.

Dans un premier temps, notre but est de rendre toutes les prédictions fausses sans se soucier de la nouvelle classe déterminée. On peut choisir de tenir compte ou non du background dans ce cas.

Pour cela, nous avons (enfin voilà quoi...) utilisé le principe du DAG (Descent Adversary Gradient). Nous considérons donc chaque pixel comme une cible, et nous cherchons à réduire le score de la classe correcte jusqu'à ce qu'il soit inférieur à celui d'une des autres classes (ie: mauvaise prédiction).

Nous créons donc successivement des perturbations sur l'image qui modifient les scores. Une fois qu'un pixel est mal classifié, il n'est plus nécessaire d'en tenir compte et on poursuit les perturbations sur le reste des pixels.

Pour choisir une perturbation efficace et peu visible, on définit une fonction de perte **L** qui vaut l'opposé de la somme des scores des classes correctement prédites.

On peut aussi générer par permutation une liste de classes cibles dont on essaye de maximiser les scores pour diriger le sens des perturbations. On tend ici vers une approche "targeted".

Notre but est donc de maximiser cette fonction **L**. On le fait par une descente de gradient qui cesse :

- lorsque tous les pixels sont mal classifiés
- ou lorsqu'un nombre maximal d'itérations a été atteint.

### Mise en oeuvre

tableau des scores en % des classes correctes:
- Pour l'image brute
- Pour l'image perturbée

set de 5 images :
- l'originale
- segmentation correcte
- la perturbation
- la nouvelle image
- segmentation fault

## Q2. Créer une attaque "targeted"

Pour contrôler l'attaque, on peut changer les conditions terminales de l'algorithme pour qu'il s'arrête lorsque les classes cibles ont toutes le meilleur score. L'aspect "targeted" est donc dans le choix de ces classes cibles.

## Q3. Evolution de la performance selon la norme autorisée de l'attaque

Faire varier gamma.

Tableau.

Il faut garder à l'esprit que les perturbations doivent rester invisibles à l'oeil nu. Il y a donc un compromis à faire entre mauvaise classification et image normale pour un humain.

## Q4. Comportement de l'attaque sur un autre réseau

Tableau comparaison scores.

On constate (oui oui) que l'attaque r entraînée sur ResNet est aussi efficace sur d'autres modèles basés sur ResNet (ayant donc une structure similaire).

ça marche aussi sur des réseaux différents mais c'est moins efficace.

On devrait pouvoir montrer que la perturbation r est efficace grâce à sa disposition spatiale (puisque faire des permutations sur les colonnes/lignes de l'image détruit l'attaque)... Ce qui favorise le fait que l'attaque soit généralisable à tout réseau.

## Q5. Comportement d'une attaque (entrainée sur plusieurs réseaux) contre 1 seul réseau

Tableau comparaison scores

Bonnes performances globales sur les réseaux utilisés pour l'entraînement, légèrement inférieures cependant au cas où on entraîne 1 réseau contre lui-même.