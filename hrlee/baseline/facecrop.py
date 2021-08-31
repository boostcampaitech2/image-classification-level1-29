import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os
from PIL import Image


if __name__ == '__main__':
    # /opt/ml/input/data/train/ 에서 실행하기

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)

    new_img_dir = 'face_images' # /opt/ml/input/data/train/face_images
    img_path = 'images' # /opt/ml/input/data/train/images
    if not os.path.isdir(new_img_dir):
        os.mkdir(new_img_dir)

    cnt = 0

    for paths in os.listdir(img_path):
        # 라벨링 에러나 숨김 파일은 건너뛰기
        label_error_ids = [
            '006359', '006360', '006361', '006362', '006363', '006364', # gender : male to female
            '001498-1', '004432', # gender : female to male
            '000020', '004418', '005227' # mask
        ]
        label_error = False
        for error_id in label_error_ids:
            if paths.startswith(error_id):
                label_error = True
                break
        if paths[0] == '.' or label_error:
            continue
        
        sub_dir = os.path.join(img_path, paths)

        for imgs in os.listdir(sub_dir):
            # 숨김 파일은 건너뛰기
            if imgs[0] == '.':
                continue
            
            # 이미지 가져오기
            img_dir = os.path.join(sub_dir, imgs)
            img = Image.open(img_dir)
            
            # mtcnn 적용
            boxes, probs = mtcnn.detect(img)
            
            # boxes 확인
            if len(probs) > 1: 
                print(boxes)
            if not isinstance(boxes, np.ndarray):
                print('Nope!')
                # 직접 crop
                #img = img[100:400, 50:350, :] # OpenCV : img[y1:y3, x1:x3]
                img = img.crop([50, 100, 350, 400]) # PIL : image.crop([x1,y1,x3,y3])
            
            # boexes size 확인
            else:
                xmin = int(boxes[0, 0])-30
                ymin = int(boxes[0, 1])-30
                xmax = int(boxes[0, 2])+30
                ymax = int(boxes[0, 3])+30
                
                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                if xmax > 384: xmax = 384
                if ymax > 512: ymax = 512
                
                #img = img[ymin:ymax, xmin:xmax, :]
                img = img.crop([xmin, ymin, xmax, ymax])
            
            tmp = os.path.join(new_img_dir, paths)
            if not os.path.isdir(tmp):
                os.mkdir(os.path.join(new_img_dir, paths))
            cnt += 1
            img.save(os.path.join(tmp, imgs))
            
    print(cnt)