import os
import sys
import json

import csv
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import vgg


def main():
    #############################results##################################
    data = [['train_loss', 'train_accurate', 'val_loss', 'val_accuracy']]
    with open('resluts.csv', 'a', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(data)
    ######################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.499, 0.264, 0.217), (0.295, 0.192, 0.163))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.499, 0.264, 0.217), (0.295, 0.192, 0.163))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "/home/zhanghuishen/Desktop/endoscope_shimulunwen/EGCGCimagev0.7"))  # get data root path
    image_path = os.path.join(data_root)  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, drop_last=True)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, drop_last=True)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=3, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 300
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            train_outputs = net(images.to(device))
            loss = loss_function(train_outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            ###################train accurate##################
            train_predict_y = torch.max(train_outputs, dim=1)[1]
            train_acc += torch.eq(train_predict_y, labels.to(device)).sum().item()
            ###################################################

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accurate = train_acc / train_num

        # validate
        net.eval()
        val_running_loss = 0.0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))

                ##########################val loss###########################
                val_loss = loss_function(outputs, val_labels.to(device))
                val_running_loss += val_loss.item()
                #############################################################

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num

        print('[epoch %d] train_loss: %.3f train_accurate: %.3f val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, val_running_loss / val_steps, val_accurate))

        ##############################results################################
        data = [[str(running_loss / train_steps), str(train_accurate), str(val_running_loss / val_steps), str(val_accurate)]]
        with open('resluts.csv', 'a', encoding='utf-8', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(data)
        #####################################################################

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net, save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
