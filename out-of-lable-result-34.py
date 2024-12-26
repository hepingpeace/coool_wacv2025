# Copyright (c) Hejiu Lu. All rights reserved.
# ---------------------------------------------
#  Modified by Hejiu Lu
# ---------------------------------------------

import torch
import numpy as np
import argparse
import pickle
import cv2
import os
import argparse
from sklearn.linear_model import LinearRegression
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

#  load files 
parser = argparse.ArgumentParser()
parser.add_argument("--annotations", type=str, default="out-of-label/annotations_public.pkl", help="Annotations Pickle File")
parser.add_argument("--video_root", type=str, default="out-of-label/video_list/",help="Folder containing video files")

args = parser.parse_args()

assert os.path.exists(args.annotations)
annotation_file = open(args.annotations, 'rb')
annotations = pickle.load(annotation_file)
annotation_file.close()

for video in annotations.keys():
    assert os.path.exists(os.path.join(args.video_root, video+".mp4"))

results_file = open("results.csv", 'w')
results_file.write("ID,Driver_State_Changed")
for i in range(23):
    results_file.write(f",Hazard_Track_{i},Hazard_Name_{i}")
results_file.write("\n")

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

for video in sorted(list(annotations.keys())):
    video_stream = cv2.VideoCapture(os.path.join(args.video_root, video+".mp4"))
    assert video_stream.isOpened()

    frame = 0
    previous_centroids = []
    previous_speeds = []
    captioned_tracks = {}
    driver_state_flag = False
    while video_stream.isOpened():
        print(f'{video}_{frame}')
        ret, frame_image = video_stream.read()
        if ret == False:
            assert frame == len(annotations[video].keys())
            break

        bboxes = []
        centroids = []
        chips = []
        track_ids = []
        for ann_type in ['challenge_object']:
            for i in range(len(annotations[video][frame][ann_type])):
                x1, y1, x2, y2 = annotations[video][frame][ann_type][i]['bbox']
                track_ids.append(annotations[video][frame][ann_type][i]['track_id'])
                bboxes.append([x1, y1, x2, y2])
                centroids.append([x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)])
                chips.append(frame_image[int(y1):int(y2), int(x1):int(x2)])
        bboxes = np.array(bboxes)
        centroids = np.array(centroids)

        if len(bboxes) == 0 or len(previous_centroids) == 0:
            frame +=1
            if len(centroids) != 0:
                previous_centroids = centroids
            continue

        # Driver state change detection
        dists = []
        for centroid in centroids:
            potential_dists = np.linalg.norm(previous_centroids - centroid, axis=1)
            min_dist = np.sort(potential_dists)[0]
            dists.append(min_dist)

        median_dist = np.median(dists)
        current_speed = median_dist  # Using median distance as a proxy for speed

        if len(previous_speeds) > 0:
            deceleration = previous_speeds[-1] - current_speed  # Calculate deceleration
            if deceleration > 5 and current_speed < 10:  # Thresholds for deceleration and speed
                driver_state_flag = True

        previous_speeds.append(current_speed)
        previous_centroids = centroids

        # Hazard detection 
        image_center = np.array([frame_image.shape[1]/2, frame_image.shape[0]/2])
        hazard_scores = []
        for i, centroid in enumerate(centroids):
            dist_to_center = np.linalg.norm(centroid - image_center)
            bbox_size = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])  # Area of bbox
            score = (1 / dist_to_center) * bbox_size  # Example scoring function
            hazard_scores.append(score)

        probable_hazard = np.argmax(hazard_scores)  # Highest score is most likely hazard
        hazard_track = track_ids[probable_hazard]

        # Hazard description
        if hazard_track not in captioned_tracks:
            hazard_chip = cv2.cvtColor(chips[probable_hazard], cv2.COLOR_BGR2RGB)
            hazard_chip = Image.fromarray(hazard_chip)
            inputs = processor(
                hazard_chip,
                return_tensors="pt"
            ).to("cuda", torch.float16)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                num_beams=3,
                no_repeat_ngram_size=2
            )

            hazard_caption = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            hazard_caption = hazard_caption.replace(","," ")
            captioned_tracks[hazard_track] = hazard_caption
        else:
            hazard_caption = captioned_tracks[hazard_track]

        # Format output for submission 
        output_line = f"{video}_{frame},{driver_state_flag},{hazard_track},{hazard_caption}"
        output_line += ", -1,  " * 22  # Fill remaining hazard slots as required
        results_file.write(output_line + '\n')

        frame += 1

results_file.close()