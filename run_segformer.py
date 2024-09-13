from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
import numpy as np
import cv2
import time
import os

device = 'cuda'

# Load the model and feature extractor (preprocessor)
model_path = "./segformer" # segformer-b5-finetuned-ade-640-640
model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
feature_extractor = SegformerImageProcessor.from_pretrained(model_path)

# Make sure the model is in eval mode
model.eval()

def text2mask(image):
    global segmentation, outputs, inputs

    # Load an image
    blob = cv2.dnn.blobFromImage(image, swapRB=True)

    # Preprocess the image
    inputs = feature_extractor(images=blob, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted segmentation map
    logits = outputs.logits  # Shape [batch_size, num_labels, height, width]
    segmentation = torch.argmax(logits, dim=1).squeeze(0)
    
    result = segmentation.cpu().numpy().astype(np.float32)
    
    segmentation = segmentation.unsqueeze(0).unsqueeze(0)
    
    class_road = 6
    class_sidewalk = 11 
    class_path = 52
    class_floor = 3
    class_dirt_track = 91
    #class_earth = 13 # v Buchlediciach sa hodi, v Piestanoch nie
    class_sand = 46
    mask = (segmentation == class_road) | (segmentation == class_sidewalk) | \
        (segmentation == class_path) | (segmentation == class_floor) | \
        (segmentation == class_dirt_track) | \
        (segmentation == class_sand)
    
    # Save the result
    mask = torch.nn.functional.interpolate(mask.float(), size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
    
    mask = mask.squeeze(0).squeeze(0)
    mask = mask.cpu().numpy().astype(np.float32)

    COLOR = [0.000, 0.447, 0.741]
    zero_mask = cv2.merge([mask*COLOR[0], mask*COLOR[1], mask*COLOR[2]])
    
    lar_valid = zero_mask>0
    masked_image = lar_valid*image
    mask_image_mix_ration = 0.65
    img_n = masked_image*mask_image_mix_ration + np.clip(zero_mask,0,1)*255*(1-mask_image_mix_ration)
    max_p = img_n.max()
    if max_p < 1e-5:
        ret = image
    else:
        img_n = 255*img_n/max_p
        ret = (~lar_valid*image)*mask_image_mix_ration + img_n
        ret = ret.astype('uint8')

    mask = (mask*255).astype(np.uint8)
    return mask, ret

if __name__ == '__main__':

    import os
    input_path = 'logs1/'
    output_path = 'outputs1-segformer/'
    name = '*' # '1724941035' # '1724940719'
    if name == '*':
        names = []
        for file_name in os.listdir(input_path):
            if file_name.lower().endswith(".png") or file_name.lower().endswith(".jpg"):
                names.append(file_name[:-4])
    else:
        names = [name]
    
    for name in names:
        
        if name == '*' and os.path.exists(output_path+name+'-output.png'):
            continue
        
        if os.path.exists(input_path+name+'.png'):
            img = cv2.imread(input_path+name+'.png')
            if img is None:
                os.rename(input_path+name+'.png',input_path+name+'.png_')
                continue
        elif os.path.exists(input_path+name+'.jpg'):
            img = cv2.imread(input_path+name+'.jpg')
            if img is None:
                os.rename(input_path+name+'.jpg',input_path+name+'.jpg_')
                continue

        img = cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
        
        t0 = time.time()

        mask, visual = text2mask(img)

        t1 = time.time()
        print(f'{name} : {t1-t0}s')

        cv2.imwrite(output_path+name+'-output.png',visual)
        #cv2.imwrite(output_path+name+'-mask.png',mask)

