# -*- coding: utf-8 -*-


import os, cv2, torch, shutil, csv
import pandas as pd
from IPython.display import display, Javascript
from google.colab import files

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Ask user choice
choice = input("Do you want to detect objects in an 'image' or a 'video'? ").strip().lower()

if choice == 'image':
    from google.colab import files
    uploaded = files.upload()
    img_path = next(iter(uploaded))  # Get uploaded image path
    img = cv2.imread(img_path)
    results = model(img)
    detected_img = results.render()[0]
    detected_path = 'detected_image.jpg'
    cv2.imwrite(detected_path, detected_img)

    # Show result
    from IPython.display import Image as IPImage
    print("Detected Image:")
    display(IPImage(detected_path))
    files.download(detected_path)

    # Save results to CSV
    csv_file = 'image_detections.csv'
    csv_data = [['object_label', 'confidence']]
    for *_, conf, cls in results.pred[0]:
        csv_data.append([model.names[int(cls)], float(conf)])
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerows(csv_data)
    display(pd.read_csv(csv_file))
    files.download(csv_file)

elif choice == 'video':
    video_path = input("Enter the path to your video (e.g. /content/video.mp4): ").strip()
    frames_dir = 'frames'
    detected_dir = 'detected_frames'
    output_video = 'output_video_with_detections.mp4'
    csv_file = 'object_counts.csv'
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(detected_dir, ignore_errors=True)
    os.makedirs(frames_dir), os.makedirs(detected_dir)

    cap = cv2.VideoCapture(video_path)
    frame_interval = 5
    frame_rate = cap.get(cv2.CAP_PROP_FPS) / frame_interval
    frame_num, csv_data = 0, [['frame', 'object_label', 'confidence']]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_num % frame_interval == 0:
            frame_name = f"frame_{frame_num:05d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_name), frame)
            results = model(frame)
            cv2.imwrite(os.path.join(detected_dir, frame_name), results.render()[0])
            for *_, conf, cls in results.pred[0]:
                csv_data.append([frame_name, model.names[int(cls)], float(conf)])
        frame_num += 1
    cap.release()

    # Save CSV
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerows(csv_data)

    # Install ffmpeg & create video
    !apt-get install -y -qq ffmpeg
    !ffmpeg -y -r {frame_rate} -pattern_type glob -i '{detected_dir}/frame_*.jpg' -c:v libx264 -pix_fmt yuv420p {output_video}

    # Display video
    display(Javascript(f"""
        var video = document.createElement('video');
        video.src = '{output_video}';
        video.controls = true;
        video.autoplay = true;
        video.loop = true;
        document.body.appendChild(video);
    """))

    # Show and download CSV + video
    display(pd.read_csv(csv_file))
    files.download(csv_file)
    files.download(output_video)

else:
    print("Invalid choice. Please type 'image' or 'video'.")
