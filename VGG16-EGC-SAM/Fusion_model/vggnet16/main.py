import os
import json
import math

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from thop import profile

# from model import vgg


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "Sensitivity", "F1", "MCC"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            Sensitivity = round(TP / (TP + FN), 3) if TN + FP != 0 else 0.
            MCC = round((TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 3) if TN + FP != 0 else 0.
            F1 = round((2 * Precision * Recall) / (Precision + Recall), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity, Sensitivity, F1, MCC])
        print(table)


    def plot(self):
        matrix = self.matrix

        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig('Confusion Matrix.jpg', dpi=None)

    def roc(self):
        #############################################ROC############################################
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(labels_all_roc[:, i], y_score_roc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr['micro'], tpr['micro'], _ = roc_curve(labels_all_roc.ravel(), y_score_roc.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        lw = 2
        plt.figure()
        plt.plot(fpr['micro'], tpr['micro'],
                 label='micro-average ROC curve (area = {0:0.3f})'
                       ''.format(roc_auc['micro']),
                 color='deeppink', linestyle='--', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'indigo','gold'])

        for i, color in zip(range(3), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.3f})'
                           ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.jpg', dpi=None)
        ##############################################################################################


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.499, 0.264, 0.217), (0.295, 0.192, 0.163))])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "/home/zhanghuishen/Desktop/endoscope_shimulunwen/EGCGCimagev0.7"))  # get data root path
    image_path = os.path.join(data_root)  # flower data set path
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform)

    batch_size = 16
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)
    # net = vgg(model_name="vgg16", num_classes=3, init_weights=True)
    # # load pretrain weights
    # model_weight_path = "./vgg16Net.pth"
    # assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net = torch.load('vgg16Net.pth')
    net.to(device)
    
    input = torch.zeros((1, 3, 224, 224)).to(device)
    flops, params = profile(net.to(device), inputs=(input,))
    print('flops', flops)
    print('params', params)


    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = net(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)


    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=3, labels=labels)
    net.eval()
    #####################ROC######################
    predict_all_roc = np.empty(shape=[0,3])
    labels_all_roc = np.empty(shape=[0,3])
    y_score_roc = np.empty(shape=[0,3])
    ##############################################
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            score = outputs = net(val_images.to(device))
            score_softmax = outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)

            ##################################ROC###################################
            labels = val_labels
            score_softmax = score_softmax.to("cpu").numpy()
            labels = labels.to("cpu").numpy()
            predic = torch.max(score.data, 1)[1].cpu().numpy()
            labels_roc = label_binarize(labels, classes=[0, 1, 2])
            labels_all_roc = np.concatenate((labels_all_roc, labels_roc), axis=0)
            y_score_roc = np.concatenate((y_score_roc,score_softmax),axis=0)
            ########################################################################

            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
    confusion.roc()






