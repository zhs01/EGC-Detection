from dataset import *
from utils import *
import function

def frames():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.507, 0.272, 0.223), (0.282, 0.190, 0.164))])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./wait_image/"))  # get data root path
    image_path = os.path.join(data_root)  # flower data set path

    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            file_path = os.path.join(root, file)
            validate_dataset.append(file_path)


    sys.path.append('./Fusion_model/vggnet16')
    net_WLI = torch.load('Fusion_model/vggnet16/vgg16Net.pth')
    net_WLI.to(device)

    args = cfg.parse_args()
    if args.dataset == 'refuge' or args.dataset == 'refuge2':
        args.data_path = './dataset/CryoPPP'

    net = get_network(args, args.net, use_gpu=True, gpu_device=device, distribution=args.distributed)

    '''load pretrained model'''
    assert args.weights != 0
    assert os.path.exists('./logs/checkpoint_best.pth')
    checkpoint_file = os.path.join('./logs/checkpoint_best.pth')
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']

    state_dict = checkpoint['state_dict']
    # state_dict = checkpoint
    if args.distributed != 'none':
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove `module.`
            name = 'module.' + k
            new_state_dict[name] = v
        # load params
    else:
        new_state_dict = state_dict

    net.load_state_dict(new_state_dict, False)

    '''segmentation data'''
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test_seg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.out_size, args.out_size)),

    ])

    # read class_indict
    json_label_path = './Fusion_model/class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')

    net.eval()
    net_WLI.eval()


    with torch.no_grad():
        for i in tqdm(range(1)):

            WLIname = validate_dataset[0]          #创建WLI数据图像

            val_images = Image.open(WLIname)
            val_images = data_transform(val_images)
            val_images = torch.unsqueeze(val_images, dim=0)   #加载WLI数据图像并transform

            outputs = net_WLI(val_images.to(device))  # 对WLI数据进行预测

            outputs = outputs.to("cpu").numpy()
            outputs_fusion = outputs
            outputs_fusion = torch.from_numpy(outputs_fusion)

            outputs_fusion = torch.argmax(outputs_fusion, dim=1)

            '''begain valuation'''
            if outputs_fusion.to("cpu").numpy() == 0:
                turkey_valid_dataset = CryopppDataset_pre(args, WLIname,
                                                                transform=transform_test,
                                                                transform_msk=transform_test_seg, mode='test',
                                                                prompt=args.prompt_approach)

                nice_test_loader = DataLoader(turkey_valid_dataset, batch_size=args.b, shuffle=False, num_workers=8,
                                              pin_memory=True)
                mask, img1, name = function.Predict_sam(args, nice_test_loader, start_epoch, net)


            elif outputs_fusion.to("cpu").numpy() == 1:
                turkey_valid_dataset = CryopppDataset_pre(args, WLIname,
                                                                transform=transform_test,
                                                                transform_msk=transform_test_seg, mode='test',
                                                                prompt=args.prompt_approach)

                nice_test_loader = DataLoader(turkey_valid_dataset, batch_size=args.b, shuffle=False, num_workers=8,
                                              pin_memory=True)
                mask, img1, name = function.Predict_sam(args, nice_test_loader, start_epoch, net)


            elif outputs_fusion.to("cpu").numpy() == 2:
                turkey_valid_dataset = CryopppDataset_pre(args, WLIname,
                                                                transform=transform_test,
                                                                transform_msk=transform_test_seg, mode='test',
                                                                prompt=args.prompt_approach)

                nice_test_loader = DataLoader(turkey_valid_dataset, batch_size=args.b, shuffle=False, num_workers=8,
                                              pin_memory=True)
                mask, img1, name = function.Predict_no_sam(args, nice_test_loader, start_epoch, net)

            img1.save('./output/' + name + '.jpg')
            mask.save('./output/' + name + '.png')

            mask_path = './output/' + name + '.png'
            image_path = './output/' + name + '.jpg'
            image = cv2.imread(image_path)
            mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_3d = mask_2d
            ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours=contours, contourIdx=-1, color=(0, 100, 0), thickness=10, lineType=cv2.LINE_AA)

            cv2.imwrite('./output/' + name + 'bl.jpg', image)

if __name__ == '__main__':
    frames()





