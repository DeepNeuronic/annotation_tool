import sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import os
import cv2
import json
import colorsys
import itertools
from pickle import load
from copy import deepcopy
from zipfile import ZipFile
from datetime import datetime
from fractions import Fraction
from argparse import Namespace
from typing import Iterable, Tuple
from resources.ByteTrack import tools
from pickle import dump, HIGHEST_PROTOCOL
from resources.utils_action_tracking import *
from shutil import rmtree, make_archive, move


###########################################################################
# CONTROL VARIABLES
###########################################################################
DEMO_TYPE = "video"
CFG_FILE_PATH = "resources/ByteTrack/exps/example/mot/yolox_x_mix_det.py"
TRAINED_MODEL_PATH = "resources/ByteTrack/pretrained/bytetrack_x_mot20.tar"
SAVE_FRAMES = False
SAVE_VIDEO = True
HSVTUPLE = Tuple[Fraction, Fraction, Fraction]
RGBTUPLE = Tuple[float, float, float]
LOGGER = None
CLASSES = [
    ["walking", "SI", "N"],
    ["running", "SI", "N"],
    ["standing", "SI", "N"],
    ["standing_up", "SI", "N"],
    ["sitting_down", "SI", "N"],
    ["jumping", "SI", "N"],
    ["riding", "PO", "N"],
    ["climbing", "PO", "N"],
    ["lying", "SI", "P"],
    ["throwing", "PO", "P"],
    ["falling", "SI", "P"],
    ["carrying_weapon", "PO", "P"],
    ["abandoned_object", "PO", "D"],
    ["fighting", "PP", "D"],
    ["stealing", "PO", "D"],
    ["shooting", "PO", "D"],
    ["vandalizing", "PO", "D"],
    ["fire_raising", "PO", "D"],
    ["fire_or_explosion", "SI", "D"],
    ["road_accident", "PP", "D"]
]


class ColourGenerator(): # class responsible for generating colours on demand

    def __init__(self):
        
        # can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
        # bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)

        self.flatten = itertools.chain.from_iterable

    def zenos_dichotomy(self) -> Iterable[Fraction]:
        
        for k in itertools.count():
            yield Fraction(1,2**k)

    def fracs(self) -> Iterable[Fraction]:
        
        yield Fraction(0)
        for k in self.zenos_dichotomy():
            i = k.denominator # [1,2,4,8,16,...]
            for j in range(1,i,2):
                yield Fraction(j,i)

    def hsvs(self) -> Iterable[HSVTUPLE]:
        
        def hue_to_tones(h: Fraction) -> Iterable[self.HSVTUPLE]:
            for s in [Fraction(6,10)]: # optionally use range
                for v in [Fraction(8,10),Fraction(5,10)]: # could use range too
                    yield (h, s, v) # use bias for v here if you use range
        
        return self.flatten(map(hue_to_tones, self.fracs()))

    def rgbs(self) -> Iterable[RGBTUPLE]:

        def hsv_to_rgb(x: HSVTUPLE) -> RGBTUPLE:
            return colorsys.hsv_to_rgb(*map(float, x))

        return map(hsv_to_rgb, self.hsvs())

    def css_colors(self) -> Iterable[str]:
        
        def rgb_to_css(x: RGBTUPLE) -> str:
            uint8tuple = map(lambda y: int(y*255), x)
            return "rgb({},{},{})".format(*uint8tuple)
        
        return map(rgb_to_css, self.rgbs())

    def colour_generator(self, num_colours):
        sample_colors = list(itertools.islice(self.css_colors(), num_colours))
        return(sample_colors)


class Logger(): # class responsible for logging every step of the annotation process
    def __init__(self):            
        
        self.message_prefix = "[" + datetime.now().strftime("%d-%b-%Y (%H:%M:%S)") + "] "

    def log(self, message):
        print(self.message_prefix + message)
    
    def show_error(self, message):

        class AnnotationException(Exception): # class responsible for raising exceptions
            def __init__(self, message):            

                super().__init__(message)

        raise AnnotationException(message = (self.message_prefix + message))


def initial_setup(): # auxiliary function, sets up the annotation process

    global LOGGER

    LOGGER = Logger()

    os.makedirs("output", exist_ok = True)

    content = os.listdir("output")

    for item in content:
        try:
            os.remove("output/" + item)
        except:
            rmtree("output/" + item)


def get_tracked_bounding_boxes(video_path): # auxiliary function, retrieves a collection of tracked bounding boxes for the target video

    LOGGER.log(message = "Computing tracked bounding boxes")

    args = {
        "demo": DEMO_TYPE,
        "exp_file": CFG_FILE_PATH,
        "ckpt": TRAINED_MODEL_PATH,
        "path": video_path,
        "expn": None,
        "name": None,
        "camid": 0,
        "save_result": True,
        "device": "gpu",
        "conf": None,
        "nms": None,
        "tsize": None,
        "fps": 30,
        "fp16": True,
        "fuse": True,
        "trt": False,
        "track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "aspect_ratio_thresh": 1.6,
        "min_box_area": 10,
        "mot20": False
    }

    args = Namespace(**args)
    
    tools.demo_track.run_demo(args = args)


def fix_tracking(remove_list, remap_list): # auxiliary function, fixes any possible error that may have been made regarding the tracking step

    if(remove_list == [] and remap_list == []): return

    LOGGER.log(message = "Fixing tracking errors")

    ############################################################
    # PERFORM AN INITIAL SETUP
    ############################################################
    with ZipFile("output/cvat_task.zip", "r") as file:
        file.extractall("output/cvat_task")

    with open("output/cvat_task/annotations.json", "r") as file:
        json_data = json.load(file)

    ############################################################################################################################
    # REMOVE AN UNWANTED ID (PARTIALLY OR COMPLETELY)
    ############################################################################################################################
    for remove_action in remove_list:
        
        if(remove_action[2] == -1): # replace the "-1" with the actual number of the last frame
            num_frames = len(list(filter(lambda x : x[0] != ".", os.listdir("output/images"))))
            remove_action[2] = num_frames
        
        for track_idx, track in enumerate(json_data[0]["tracks"]):
            if(track["attributes"][0]["value"] == remove_action[0]): # we have found the ID from whom we want to remove frames
                removed_track_idx = track_idx
                
                for frame_to_remove in range(remove_action[1], remove_action[2] + 1):
                    frame_to_remove -= 1

                    # -------------------------------------------------------------------------------------
                    # find the index of the shape (i.e., the bounding box information) we want to eliminate
                    # -------------------------------------------------------------------------------------
                    shape_index = None
                    for shape_idx, shape in enumerate(json_data[0]["tracks"][removed_track_idx]["shapes"]):
                        if(shape["frame"] == frame_to_remove): shape_index = shape_idx
                    
                    if(shape_index is None): continue

                    json_data[0]["tracks"][removed_track_idx]["shapes"].pop(shape_index)
        
        if(json_data[0]["tracks"][removed_track_idx]["shapes"] == []): # this ID has no occurrences left, we should eliminate it
            json_data[0]["tracks"].pop(removed_track_idx)

    #####################################################################################################################################################
    # REMAP ONE ID TO ANOTHER (PARTIALLY OR COMPLETELY)
    #####################################################################################################################################################
    for remap_action in remap_list:
        
        if(remap_action[2] == -1): # replace the "-1" with the actual number of the last frame
            num_frames = len(list(filter(lambda x : x[0] != ".", os.listdir("output/images"))))
            remap_action[2] = num_frames
        
        # find the index of the target ID
        target_track_idx = None
        for track_idx, track in enumerate(json_data[0]["tracks"]):
            if(track["attributes"][0]["value"] == remap_action[-1]): # we have found the ID from whom we want to remap frames
                target_track_idx = track_idx

        if(target_track_idx is None): # the target ID does not exist, so let's create it
            new_track = {"frame":0,"group":0,"source":"manual","shapes":[], "attributes":[{"value": remap_action[-1],"name":"id"}],"label":"human"}
            json_data[0]["tracks"].append(new_track)
            target_track_idx = -1

        for track_idx, track in enumerate(json_data[0]["tracks"]):
            if(track["attributes"][0]["value"] == remap_action[0]): # we have found the ID from whom we want to remap frames
                remaped_track_idx = track_idx

                for frame_to_remap in range(remap_action[1], remap_action[2] + 1):
                    frame_to_remap -= 1

                    # -------------------------------------------------------------------------------------
                    # find the index of the shape (i.e., the bounding box information) we want to eliminate
                    # -------------------------------------------------------------------------------------
                    shape_index = None
                    for shape_idx, shape in enumerate(json_data[0]["tracks"][remaped_track_idx]["shapes"]):
                        if(shape["frame"] == frame_to_remap): shape_index = shape_idx
                    
                    if(shape_index is None): continue

                    # ---------------------------------------------------------------------------------------------------------------------------------
                    # remap the bounding box to the target ID
                    # ---------------------------------------------------------------------------------------------------------------------------------
                    json_data[0]["tracks"][target_track_idx]["shapes"].append(deepcopy(json_data[0]["tracks"][remaped_track_idx]["shapes"][shape_index]))
                    json_data[0]["tracks"][remaped_track_idx]["shapes"].pop(shape_index)
        
        if(json_data[0]["tracks"][remaped_track_idx]["shapes"] == []): # this ID has no occurrences left, we should eliminate it
            json_data[0]["tracks"].pop(remaped_track_idx)

    # save the updates made to the tracking step
    with open("output/cvat_task/annotations.json", "w") as file:
        json.dump(json_data, file)

    ###################################################################################################################################################################################
    # VISUALIZE THE UPDATES TO THE TRACKING STEP
    ###################################################################################################################################################################################
    #rmtree("output/tracked_objects")
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # save each object's tracking
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for track in json_data[0]["tracks"]:
        os.makedirs("output/tracked_objects_fixed/" + str(track["attributes"][0]["value"]), exist_ok = True)

        for occurrence in track["shapes"]:
            bbox = list(map(lambda x : int(x), occurrence["points"]))

            frame_np = np.asarray(Image.open("output/images/{:06d}.jpg".format(int(occurrence["frame"]) + 1)))

            cropped = frame_np[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
            Image.fromarray(cropped.astype(np.uint8)).save("output/tracked_objects_fixed/" + str(track["attributes"][0]["value"]) + "/{:06d}.jpg".format(int(occurrence["frame"]) + 1))

    ###########################################################
    # FINALIZE EVERYTHING
    ###########################################################
    make_archive("output/cvat_task", 'zip', "output/cvat_task")
    rmtree("output/cvat_task")


def annotate_actions(video_path, actions_list): # auxiliary function, annotates the actions that each person is doing

    if(actions_list == []): LOGGER.show_error(message = "Please add action annotations!")

    LOGGER.log(message = "Annotating actions")

    def get_class_instance(frame_number, object_id, actions_list): # auxiliary function, returns the class instance performed by "object_id" in frame "frame_number"
        
        for action_annotation in actions_list:
            if((action_annotation[0] == object_id) and (frame_number >= action_annotation[1]) and (frame_number <= action_annotation[2])):
                return(action_annotation[3] + "_" + str(object_id))
        
        LOGGER.show_error(message = "Error! (One of the action annotations is invalid)")

    #################################################################################################################
    # PERFORM AN INITIAL SETUP
    #################################################################################################################
    annotations_dict = {}

    video = cv2.VideoCapture(video_path)

    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    annotations_dict[video_path.split("/")[-1]] = [{}, width, height, total_frames, fps, "-"]

    with ZipFile("output/cvat_task.zip", "r") as file:
        file.extractall("output/cvat_task")

    with open("output/cvat_task/annotations.json", "r") as file:
        json_data = json.load(file)

    for action_annotation_idx in range(len(actions_list)):
        if(actions_list[action_annotation_idx][2] == -1): # replace the "-1" with the actual number of the last frame
            num_frames = len(list(filter(lambda x : x[0] != ".", os.listdir("output/images"))))
            actions_list[action_annotation_idx][2] = num_frames

    #############################################################################################################################################################
    # MERGE THE TRACKING INFORMATION WITH THE ACTION ANNOTATIONS
    #############################################################################################################################################################
    for track in json_data[0]["tracks"]:
        for shape in track["shapes"]:
            frame_number = shape["frame"] + 1
            bbox = [int(shape["points"][0]), int(shape["points"][1]), int(shape["points"][2] - shape["points"][0]), int(shape["points"][3] - shape["points"][1])]
            class_instance = get_class_instance(frame_number = frame_number, object_id = track["attributes"][0]["value"], actions_list = actions_list)

            if(frame_number not in annotations_dict[video_path.split("/")[-1]][0].keys()):
                annotations_dict[video_path.split("/")[-1]][0][frame_number] = {}

            annotations_dict[video_path.split("/")[-1]][0][frame_number][class_instance] = bbox

    with open("output/" + video_path.split("/")[-1] + ".pkl", "wb") as file:
        dump(annotations_dict, file, HIGHEST_PROTOCOL)


def extract_frames(video_path): # auxiliary function, extracts images from the target video
    
    LOGGER.log(message = "Extracting frames")

    os.makedirs("output/images", exist_ok = True)

    video = cv2.VideoCapture(video_path)
    frame_counter = 0

    while(True):
        
        (grabbed, frame) = video.read()
        if(not grabbed): break
        frame_counter += 1
        
        cv2.imwrite("output/images/{:06d}.jpg".format(frame_counter), frame)
        
    video.release()


def visualize_annotations(video_path): # auxiliary function, visualizes the final annotations on the target video

    ##########################################################################################
    # PERFORM AN INITIAL SETUP
    ##########################################################################################
    LOGGER.log(message = "Visualizing the final annotations")

    # load the annotations
    with open("output/" + video_path.split("/")[-1] + ".pkl", "rb") as file:
        annotations_dict = load(file)

    annotations_dict = annotations_dict[list(annotations_dict.keys())[0]][0]

    dangerous_captions = []
    for caption in CLASSES:
        if(caption[2] == "D"): dangerous_captions.append(caption[0].replace(" COMMA ", ", "))

    pre_defined_colors = {"N": (153, 153, 153), "P": (255, 117, 26), "D": (230, 0, 0)}

    color_maps = {}

    for i in CLASSES:
        color_maps[i[0].replace("_COMMA_", ", ").replace("_", " ")] = pre_defined_colors[i[2]]

    video = cv2.VideoCapture(video_path)
    if(SAVE_VIDEO): writer = None

    og_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    og_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps = video.get(cv2.CAP_PROP_FPS)

    if(og_height < 720):
        target_short_size = 720
        target_size = (int((og_width * target_short_size) / og_height), target_short_size)

    else:
        target_size = (og_width, og_height)

    ###################################################################################################################################################################################
    # ACTUALLY VISUALIZE THE ANNOTATIONS
    ###################################################################################################################################################################################
    frame_counter = 1

    while(True):
        
        (grabbed, frame) = video.read()
        
        if(not grabbed): break

        frame = cv2.resize(frame, target_size)

        if(frame_counter in annotations_dict.keys()):
            last_visual_mask = last_visual_mask = visual_result(annotations_dict[frame_counter], (og_width, og_height), frame.shape[1], frame.shape[0], color_maps, dangerous_captions)
            frame = visual_frame(frame, last_visual_mask)

        if(SAVE_VIDEO):
            if writer is None:
                # initialize the video writer
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                #writer = cv2.VideoWriter("results/final_annotations.mp4", fourcc, fps, (frame.shape[1], frame.shape[0]), True)
                writer = cv2.VideoWriter("output/" + video_path.split("/")[-1], fourcc, fps, (frame.shape[1], frame.shape[0]), True)

        if(SAVE_FRAMES):
            if(frame_counter in annotations_dict.keys()):                
                os.makedirs("output/final_annotated_frames", exist_ok = True)
                cv2.imwrite("output/final_annotated_frames/%d.jpg" % frame_counter, frame)
        
        # if requested, write the output frame to disk
        if(SAVE_VIDEO): writer.write(frame)

        frame_counter += 1

    if(SAVE_VIDEO): writer.release()
    video.release()


def finalize(video_path): # auxiliary function, finalizes the annotation process
    
    LOGGER.log(message = "Finalizing the annotation process")

    os.makedirs("output/" + video_path.split("/")[-1].split(".")[0], exist_ok = True)

    if(os.path.exists("output/cvat_task")): rmtree("output/cvat_task")
    if(os.path.exists("output/tracked_objects")): rmtree("output/tracked_objects")
    if(os.path.exists("output/tracked_objects_fixed")): rmtree("output/tracked_objects_fixed")

    os.rename("output/cvat_task.zip", "output/" + video_path.split("/")[-1].split(".")[0] + "/cvat_task.zip")
    if(os.path.exists("output/" + video_path.split("/")[-1])): os.rename("output/" + video_path.split("/")[-1], "output/" + video_path.split("/")[-1].split(".")[0] + "/" + video_path.split("/")[-1])
    os.rename("output/" + video_path.split("/")[-1] + ".pkl", "output/" + video_path.split("/")[-1].split(".")[0] + "/" + video_path.split("/")[-1] + ".pkl")
    move("output/images", "output/" + video_path.split("/")[-1].split(".")[0] + "/images")

    make_archive("output/" + video_path.split("/")[-1].split(".")[0], 'zip', "output/" + video_path.split("/")[-1].split(".")[0])
    rmtree("output/" + video_path.split("/")[-1].split(".")[0])