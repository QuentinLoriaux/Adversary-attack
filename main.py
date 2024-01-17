import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np

#NOTE : il faudra faire ça avec une 50aine d'img selon les consignes

#images sous forme de tenseurs flottants avec pixels normalisés sur [0,1]
im1 = torchvision.io.read_image("./coco/217730183_8f58409e7c_z.jpg").float()/255
im2 = torchvision.io.read_image("./coco/541870527_8fe599ec04_z.jpg").float()/255
im3 = torchvision.io.read_image("./coco/2124681469_7ee4868747_z.jpg",mode=torchvision.io.ImageReadMode.RGB).float()/255
im4 = torchvision.io.read_image("./coco/2711568708_89f2308b85_z.jpg").float()/255
im5 = torchvision.io.read_image("./coco/2928196999_acd5471d23_z.jpg").float()/255
im6 = torchvision.io.read_image("./coco/3016145160_497da1b387_z.jpg").float()/255
im7 = torchvision.io.read_image("./coco/4683642953_2eeda0820e_z.jpg").float()/255
im8 = torchvision.io.read_image("./coco/6911037487_cc68a9d5a4_z.jpg").float()/255
im9 = torchvision.io.read_image("./coco/8139728801_60c233660e_z.jpg").float()/255

#resize en carrés de 520x520px
im1 = torch.nn.functional.interpolate(im1.unsqueeze(0), size=520)[0]
im2 = torch.nn.functional.interpolate(im2.unsqueeze(0), size=520)[0]
im3 = torch.nn.functional.interpolate(im3.unsqueeze(0), size=520)[0]
im4 = torch.nn.functional.interpolate(im4.unsqueeze(0), size=520)[0]
im5 = torch.nn.functional.interpolate(im5.unsqueeze(0), size=520)[0]
im6 = torch.nn.functional.interpolate(im6.unsqueeze(0), size=520)[0]
im7 = torch.nn.functional.interpolate(im7.unsqueeze(0), size=520)[0]
im8 = torch.nn.functional.interpolate(im8.unsqueeze(0), size=520)[0]
im9 = torch.nn.functional.interpolate(im9.unsqueeze(0), size=520)[0]


# ============ Prédiction correcte ============

W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W)
net = net.cuda()

with torch.no_grad():
  x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
  x = (W.transforms())(x).cuda()
  z = net(x)["out"] # prédiction des cartes de score de confiance
  classes = [0,8,12,15,    16,3,2,4,5]
  z = z[:,classes,:,:] # we keep only background, person, cat and dog class + wrong classes
   #exemple : z[:,0,:,:] donne le score du background sur chaque pixel
  _,l = z.max(1) 
  del z


# ============ Bagarre ============
#Je tente le DAG (Dense Adversary Generation), voir [p.3] Xie_Adversarial_Examples_for_ICCV_2017_paper.pdf



#untargeted correspondrait à une descente de gradient où on fait baisser le bon score jusqu'à ce que ce soit faux?
#ou alors c'est avec une permutation aléatoire

#définir la liste des mauvaises prédictions (par permutation parmi les mauvaises par ex, ou choix arbitraire)

n1,n2,n3 = np.shape(l)
l_prime = np.empty((n1, n2, n3), dtype=int)

def permutation_sans_point_fixe(lst):
    while True:
        perm = np.random.permutation(lst)
        if all(x != y for x, y in zip(lst, perm)):
            return perm

perm = permutation_sans_point_fixe(range(len(classes)))

for i in range(n1):
    for j in range(n2):
        for k in range(n3):
            l_prime[i, j, k] = perm[l[i,j,k]]

gamma = 0.5
maxIter = 200

# x : base image
# l : original label
# l_prime : adversarial label
# gamma : paramètre d'intensité de la perturbation


xm = x
lm = l
r = torch.zeros_like(x)
m = 0

print("debut boucle")

while m<maxIter and torch.any(torch.eq(l,lm)) : 

  #calcul sur l'image complète
  with torch.no_grad():
    _,lmm=net(xm)["out"][:,classes,:,:].max(1)
  


  masque = (lmm == l).unsqueeze(1).expand(-1, 3, -1, -1)
  print(masque.shape)
  print(xm.shape)
  xm_s = torch.masked_select(xm,masque).cpu() #On masque les pixels déjà mal classés
  # pas assez de mem sur le gpu, je tente le cpu au pif

  #calcul sur l'image réduite
  zm_s = net(xm_s)["out"]
  zm_s = zm_s[:,classes,:,:].requires_grad_()

  zm_s.backward()
  grads = zm_s.grad

  n1m, _, n2m, n3m = zm_s.shape

  #calcul de la perturbation
  rm = torch.zeros_like(xm)
  rm_s =  torch.masked_select(rm,masque) 
  
  for i in range(n1m):
        for j in range(n2m):
            for k in range(n3m):
              if lmm[i,j,k] == l[i,j,k] : # on ne traite que les pixels correctement classifiés
                  rm_s[i,j,k]  = rm_s[i,j,k] + grads[i,l_prime[i,j,k],j,k] - grads[i,l[i,j,k],j,k]

  rm_s = (gamma/torch.norm(rm))*rm_s
  rm[masque] = rm_s

  r = r + rm
  xm = xm + rm
  m = m+1
  print(m)

# r: perturbation finale
# on a entrainé notre attaque, si on prend une image X, X+r correspond à l'image ayant subi l'attaque
# ça pose un pitit problème de dimensions si X a pas les bonnes dims... à mon avis

# l'intérêt dans la suite sera de tester net(X+r) avec plusieurs réseaux (celui qui a servi d'entrainement mais aussi d'autres)
# et de faire des tableaux comparatifs de résultats

# Pour ce qui est d'apprendre sur plusieurs réseaux, j'imagine qu'on continue la boucle jusqu'à ce que toutes les classes soient
# fausses selon les critères de chaque réseau
# lm[...] == l[...] or l2m[...] == l2[...] or ...






# une attaque "targeted" correspondrait peut-être à faire apparaître complètement une autre forme + autre classe dans l'image
# par exemple? 




x = x + r

# a priori, y a moyen que des groupes se séparent et que des formes changent... ça fait plus de classes
# à afficher!

# ============ Affichage ============

# visualisation des prédictions : il faut transformer les indices de classes en couleur
class_to_color = {0: [120, 120, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [0, 0, 255], 5: [0, 0, 255], 6: [0, 0, 255], 7: [0, 0, 255], 8: [0, 0, 255], 9: [0, 0, 255]}
# Dans l'ordre de découpage du tenseur ligne 40
# 0 : background
# 1 : humain
# 2 : chat
# 3 : 

rgb_tensor = torch.zeros(9,3,520,520).cuda()
for class_label, color in class_to_color.items():
    mask = (z == class_label).unsqueeze(1).float()
    rgb_tensor += mask * torch.tensor(color).view(1, 3, 1, 1).float().cuda()

visu = torch.cat([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=-1)
visubis = torch.cat([rgb_tensor[i] for i in range(9)],dim=-1).cpu()
visu = torch.cat([visu,visubis],dim=1)
visu = visu.cpu().numpy().transpose(1,2,0)

dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 400
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
plt.show()