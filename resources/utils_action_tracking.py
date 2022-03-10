import sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import cv2
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

def get_action(frame, tracked_object, action_list): # auxiliary function, retrieves the action that a tracked object is doing in a given frame

    for i in action_list:
        if(i[2] == -1): i[2] = sys.maxsize

        if((i[0] == tracked_object) and (frame >= i[1]) and (frame <= i[2])):
            return(i[3])

    return("error")

def resize_image(image, width = None, height = None, inter = cv2.INTER_AREA): # auxiliary function, resizes the given image while keeping its original aspect ratio
    
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if(width is None and height is None):
        return(image)

    # check to see if the width is None
    if(width is None):
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return(resized)

def intersection_over_union(boxA, boxB): # auxiliary function, computes the IoU between two bounding boxes
    
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
    
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
	iou = interArea / float(boxAArea + boxBArea - interArea)
    
	return(iou)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask=None): # auxiliary function, overlays one image on top of another
    
    if y < 0 or y + img_overlay.shape[0] > img.shape[0] or x < 0 or x + img_overlay.shape[1] > img.shape[1]:
        y_origin = 0 if y > 0 else -y
        y_end = img_overlay.shape[0] if y < 0 else min(img.shape[0] - y, img_overlay.shape[0])

        x_origin = 0 if x > 0 else -x
        x_end = img_overlay.shape[1] if x < 0 else min(img.shape[1] - x, img_overlay.shape[1])

        img_overlay_crop = img_overlay[y_origin:y_end, x_origin:x_end]
        alpha = alpha_mask[y_origin:y_end, x_origin:x_end] if alpha_mask is not None else None
    else:
        img_overlay_crop = img_overlay
        alpha = alpha_mask

    y1 = max(y, 0)
    y2 = min(img.shape[0], y1 + img_overlay_crop.shape[0])

    x1 = max(x, 0)
    x2 = min(img.shape[1], x1 + img_overlay_crop.shape[1])

    img_crop = img[y1:y2, x1:x2]
    img_crop[:] = alpha * img_overlay_crop + (1.0 - alpha) * img_crop if alpha is not None else img_overlay_crop


def visual_result(predictions, og_size, width, height, color_maps, dangerous_captions): # auxiliary function, creates the bounding box(es) and corresponding label(s)

    normal_colour = (51, 51, 51)
    dangerous_colour = (179, 0, 0)

    predictions = deepcopy(predictions)

    # rescale the predictions
    for k, v in predictions.items():

        box = [
            int((v[0] / og_size[0]) * width),
            int((v[1] / og_size[1]) * height),
            int((v[2] / og_size[0]) * width),
            int((v[3] / og_size[1]) * height)
        ]

        predictions[k] = box

    long_side = min(width, height)
    font_size = max(int(round((long_side / 60))), 1)
    box_width = max(int(round(long_side / 180)), 1)
    font = ImageFont.truetype("resources/roboto/RobotoCondensed-Regular.ttf", font_size)

    result_vis = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(result_vis)

    if(predictions is None): return(result_vis)
    
    boxes_and_captions = {}
    
    bboxes_colors = {"_".join(list(map(lambda x : str(x), v))): normal_colour for k, v in predictions.items()}
    
    bboxes_colors = {}
    for k, v in predictions.items():

        bboxes_colors["_".join(list(map(lambda x : str(x), v)))] = color_maps[" ".join(k.split("_")[:-1])]
    
    drawn_bboxes = {i: False for i in bboxes_colors.keys()}

    for caption, box in predictions.items():

        bbox_string = "_".join(list(map(lambda x : str(x), box)))

        if(not drawn_bboxes[bbox_string] or bboxes_colors[bbox_string] == dangerous_colour):
                
            draw.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline = bboxes_colors[bbox_string], width = box_width)
            drawn_bboxes[bbox_string] = True

        if(bbox_string not in boxes_and_captions.keys()):
            boxes_and_captions[bbox_string] = [caption]
        else:
            boxes_and_captions[bbox_string] += [caption]
    
    for box, captions in boxes_and_captions.items():

        box = list(map(lambda x : int(x), box.split("_")))
        captions = [caption.replace("_", " ").replace(" COMMA ", ", ") for caption in captions]
        
        if len(captions) == 0: continue

        x1, y1, x2, y2 = box
        overlay = Image.new("RGBA", result_vis.size, (0, 0, 0, 0))
        trans_draw = ImageDraw.Draw(overlay)
        caption_sizes = [trans_draw.textsize(caption, font = font) for caption in captions]
        caption_widths, caption_heights = list(zip(*caption_sizes))
        max_height = max(caption_heights)
        rec_height = int(round(1.8 * max_height))
        space_height = int(round(0.2 * max_height))
        total_height = (rec_height + space_height) * (len(captions) - 1) + rec_height
        width_pad = max(font_size // 2, 1)
        start_y = max(round(y1) - total_height, space_height) - 1

        for i, caption in enumerate(captions):
            
            if((" ".join(caption.split(" ")[:-1])) not in color_maps): continue

            background_color = color_maps[" ".join(caption.split(" ")[:-1])]
            r_x1 = round(x1)
            r_y1 = start_y + (rec_height + space_height) * i
            r_x2 = r_x1 + caption_widths[i] + width_pad * 2
            r_y2 = r_y1 + rec_height
            rec_pos = (r_x1, r_y1, r_x2, r_y2)

            height_pad = round((rec_height - caption_heights[i]) / 2)
            text_pos = (r_x1 + width_pad, r_y1 + height_pad)

            trans_draw.rectangle(rec_pos, fill = background_color)
            trans_draw.text(text_pos, caption, fill = (255, 255, 255, 255), font = font, align = "center")

        result_vis = Image.alpha_composite(result_vis, overlay)

    return result_vis

def visual_frame(frame, visual_mask): # auxiliary function, overlays the predictions / annotations on top of the source frame

    img = Image.fromarray(frame[..., ::-1])
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, visual_mask)

    img = img.convert("RGB")

    return np.array(img)[..., ::-1]