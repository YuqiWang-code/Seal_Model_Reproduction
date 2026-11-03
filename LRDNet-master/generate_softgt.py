import os
import cv2
from dataset.dataloader import GenerateDataset
from tqdm import tqdm
from config import param as option
from model.model import get_model
from utils import set_seed

if __name__ == '__main__':
    set_seed(option)
    model = get_model(option)
    model.eval()

    result_save_path = option['soft_gt_root']
    os.makedirs(result_save_path, exist_ok=True)
    generate_root = option['paths']['image_root']
    generate_loader = GenerateDataset(generate_root, option['trainsize'])

    for i in tqdm(range(generate_loader.size)):
        image, HH, WW, name = generate_loader.load_data()
        name = os.path.splitext(name)[0] + '.png'
        image = image.cuda()
        res, _ = model.forward(image, (HH,WW))
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(os.path.join(result_save_path, name), res)

    print('Successful, generate in {}'.format(result_save_path))



