import segmentation_models_pytorch as smp
from PIL import Image 
import matplotlib.pyplot as plt 
import torch
import torchvision
import torchvision.transforms as T
import numpy as np

# caricamento modello
model = smp.DeepLabV3Plus(
    encoder_name = "resnet101",
    encoder_weights = "imagenet"
)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# definisco la funzione di pre-processing sull'input e la applico sull'immagine di prova. Questa:
# 1) Ridimensiona l'immagine a 256x256
# 2) Fa un Centercrop a (224x224)
# 3) Converte in tensore, i valori vengono scalati per stare nell'intervallo [0,1] anziché [0,255]
# 4) Normalizza con i valori di mean e std di ImageNet inizializzati sopra
# 5) Aggiunge una dimensione per rendere il tensore [1 x C x H x W], questo serve perché abbiamo bisogno di un batch.

trf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
    T.PILToTensor()
])

cs_data = torchvision.datasets.Cityscapes(root='Dataset',split='train',mode='fine',transform=trf,
                                                target_type='semantic')
data_loader = torch.utils.data.DataLoader(cs_data,
                                          batch_size=4,
                                          shuffle=True)

def decode_segmap(image, nc=21):
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb

def segment(net, img):
  inp = img.unsqueeze(0)
  out = net(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  rgb = decode_segmap(om)
  plt.imshow(rgb); plt.axis('off'); plt.show()

for img_batch, smnt_batch in data_loader:
  model(img_batch)
