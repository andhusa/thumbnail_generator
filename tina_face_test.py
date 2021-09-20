import torch
path = "/home/andrehus/egne_prosjekter/videoAndOutput/models/face_detection/tinaface_r50_fpn_gn_dcn.pth"

model = torch.load(path)
model.eval()