import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np

#NOTE : il faudra faire ça avec une 50aine d'img selon les consignes
#NOTE : Ici, on enlèvera les commentaires, on fera du code + propre et avec des fonctions
#       de manière à ce qu'on puisse facilement faire des tests
#       Faut implémenter d'autres réseaux aussi
#NOTE : Il faut faire de quoi sauvegarder les résultats sous forme d'image, et stocker les perturbations créées!

def preprocess_images(file_paths, size=520):
    images = []
    for file_path in file_paths:
        image = torchvision.io.read_image(file_path,mode=torchvision.io.ImageReadMode.RGB).float() / 255
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size)[0]
        images.append(image)
    return images

image_file_paths = [
    "./coco/217730183_8f58409e7c_z.jpg",
    "./coco/541870527_8fe599ec04_z.jpg",
    "./coco/2124681469_7ee4868747_z.jpg",
    "./coco/2711568708_89f2308b85_z.jpg",
    "./coco/2928196999_acd5471d23_z.jpg",
    "./coco/3016145160_497da1b387_z.jpg",
    "./coco/4683642953_2eeda0820e_z.jpg",
    "./coco/6911037487_cc68a9d5a4_z.jpg",
    "./coco/8139728801_60c233660e_z.jpg",
]

imgs = preprocess_images(image_file_paths)


# ============ Prédiction correcte ============

W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W)
net = net.cuda()

classes = [0,8,12,15,    16,3,2,4,5]
nb_imgs = min(2, len(image_file_paths))
with torch.no_grad():
  x = torch.stack(imgs[:nb_imgs],dim=0)
  x = (W.transforms())(x).cuda() # ajuste notamment à 520 la taille
  _,l = net(x)["out"][:,classes,:,:].max(1) # prédiction des cartes de score de confiance


# ============ Attaque avec DAG (Dense Adversary Generation) ============

#fonction de perte
class adversarialLoss(torch.nn.Module):
    def __init__(self):
        super(adversarialLoss, self).__init__()

    def forward(self, z, condition, l, l_prime):
        # Sélectionner les indices où condition est True
        indices = torch.where(condition)
        z_good = z[indices[0], l[indices], indices[1], indices[2]]
        z_bad = z[indices[0], l_prime[indices], indices[1], indices[2]]

        loss = torch.sum(z_good - z_bad)

        return loss


#définir la liste des mauvaises prédictions
def permutation_sans_point_fixe(l, classes):
    lst = range(len(classes))
    perm = np.random.permutation(lst)
    while not all(x != y for x, y in zip(lst, perm)):
        perm = np.random.permutation(lst)
    
    value_mapping = {value: index for index, value in enumerate(perm)}
    with torch.no_grad():
        mapping_tensor = torch.tensor([value_mapping[val] for val in range(l.max() + 1)], dtype=l.dtype)
        l_prime = mapping_tensor[l.cpu()]
        return l_prime





def attack(net, x, l, gamma = 0.8, maxIter = 10):
    xm = x.detach().requires_grad_()
    with torch.no_grad():
        l_prime = permutation_sans_point_fixe(l.cpu(), classes)
        r = torch.zeros_like(x)
    m = 0

    net.cpu()
    net.zero_grad()
    print("debut boucle")

    while m < maxIter: 
        m += 1
        print("tour : " + str(m))
        
        zm = net(xm.cpu())["out"][:,classes,:,:].cpu().requires_grad_()
        
        with torch.no_grad():
            _,lm = zm.max(1)
            condition = torch.eq(lm.cuda(), l.cuda())
            somme = condition.sum()
            print("pixels correctement classés : " + str(somme.item()))
            
            if condition.sum() == 0: # tous les pixels sont mal classés : fin de l'algorithme
                break
                
        # print("a")
        loss = adversarialLoss()
        # print("b")
        loss = loss(zm.cuda(), condition.cuda(), l.cuda(), l_prime.cuda())
        # print("c")
        loss.backward(retain_graph=False)
        # print("d")
        grad = xm.grad
        # print("e")
        rm = torch.zeros_like(x)

        indices = torch.where(condition)
        rm[indices[0],:,indices[1],indices[2]] = -grad[indices[0],:,indices[1],indices[2]]

        rm = (gamma/rm.norm())*rm
        r += rm


        xm=xm.detach()
        xm += rm
        xm.grad = None
        xm.requires_grad_()
        
        del zm
        net.zero_grad()

    return r

r = attack(net, x, l)
x = x + r
_,l = net(x.cpu())["out"][:,classes,:,:].max(1)

# a priori, y a moyen que des groupes se séparent et que des formes changent... ça fait plus de classes
# à afficher!

# ============ Affichage ============

def display(x, r, l):


    # visualisation des prédictions : il faut transformer les indices de classes en couleur
    class_to_color = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [0, 0, 255], 5: [0, 0, 255], 6: [0, 0, 255], 7: [0, 0, 255], 8: [0, 0, 255], 9: [0, 0, 255]}
    # Dans l'ordre de découpage du tenseur ligne 40
    # 0 : background
    # 1 : humain
    # 2 : chat
    # 3 : 

    rgb_tensor = torch.zeros(x.shape)
    for class_label, color in class_to_color.items():
        mask = (l == class_label).unsqueeze(1).float()
        rgb_tensor += mask * torch.tensor(color).view(1, 3, 1, 1).float()

    #mettre le bon nombre d'images
    visu = torch.cat(imgs[:nb_imgs],dim=-1)
    print(visu.shape)
    visubis = torch.cat([rgb_tensor[i] for i in range(x.size(0))],dim=-1).cpu()
    print(visubis.shape)
    visu = torch.cat([visu,visubis],dim=1)
    visu = visu.cpu().numpy().transpose(1,2,0)

    dpi = plt.rcParams['figure.dpi']
    width_px = 1600
    height_px = 400
    plt.figure(figsize=(width_px/dpi, height_px/dpi))
    plt.imshow(visu)
    plt.axis('off')
    plt.show()