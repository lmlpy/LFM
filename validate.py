import os.path
import time
import torch
import numpy as np
from networks.LFMSN import LFMSN
from sklearn.metrics import average_precision_score, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader

def validate(model, opt):
    data_loader = create_dataloader(opt)
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens)[0].sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

def validate_feature(model, opt):
    data_loader = create_dataloader(opt)
    index = 0
    with torch.no_grad():
        for img, label in data_loader:
            in_tens = img.cuda()
            features = model.encoder(in_tens)
            for i in range(features.shape[0]):
                feature = features[i].reshape(-1)
                if not os.path.exists(os.path.join('./feature_show/Avg',opt.dataroot[31:-1],str(int(label[i])))):
                    os.makedirs(os.path.join('./feature_show/Avg',opt.dataroot[31:-1],str(int(label[i]))))
                path = os.path.join('./feature_show/Avg',opt.dataroot[31:-1],str(int(label[i])),str(index)+'.pth')
                torch.save(feature,path)
                index += 1
    return 0, 0, 0, 0, 0, 0

def validate_loss(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = LFMSN()
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
