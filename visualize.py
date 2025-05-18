import ast

import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


results = pd.read_csv('./test_interpolated.csv')

# load video
video_path = 'vid.mp4'  # Updated to match the actual file in the directory
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {video_path}")
    exit(1)

# Print video information for debugging
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video loaded: {video_path} | Total frames: {total_frames} | FPS: {fps}")


# Try different codecs - H.264 is more compatible on Windows
try:
    # First try H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_path = './out.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Test if VideoWriter is initialized properly
    if not out.isOpened():
        # Fall back to XVID codec which has better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = './out.avi'  # AVI container for XVID
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("Could not initialize video writer with any codec")
        else:
            print(f"Using XVID codec with output: {output_path}")
    else:
        print(f"Using H264 codec with output: {output_path}")
except Exception as e:
    print(f"Error setting up video writer: {e}")
    cap.release()
    exit(1)

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[ERROR] Failed to read frame for car_id={car_id}. Skipping this car.")
        continue  # skip processing for this car_id

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                # Calculate target positions
                y_start = max(0, int(car_y1) - H - 100)
                y_end = max(0, int(car_y1) - 100)
                x_start = max(0, int((car_x2 + car_x1 - W) / 2))
                x_end = min(frame.shape[1], int((car_x2 + car_x1 + W) / 2))
                
                # Check if dimensions match before assignment
                target_height = y_end - y_start
                target_width = x_end - x_start
                
                if target_height > 0 and target_width > 0:
                    # Resize license crop to match the target dimensions
                    resized_crop = cv2.resize(license_crop, (target_width, target_height))
                    # Overlay the license plate
                    frame[y_start:y_end, x_start:x_end, :] = resized_crop
                
                # Calculate white background positions for text
                bg_y_start = max(0, int(car_y1) - H - 400)
                bg_y_end = max(0, int(car_y1) - H - 100)
                bg_width = x_end - x_start
                
                if bg_y_end > bg_y_start and bg_width > 0:
                    # Create white background of the correct size
                    white_bg = np.ones((bg_y_end - bg_y_start, bg_width, 3), dtype=np.uint8) * 255
                    # Overlay the white background
                    frame[bg_y_start:bg_y_end, x_start:x_end, :] = white_bg

                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except Exception as e:
                print(f"Error processing license plate: {e}")

        # Write frame to output video
        try:
            out.write(frame)
            # Resize for display (if needed)
            frame_display = cv2.resize(frame, (1280, 720))
            
            # Uncomment to display frames while processing
            # cv2.imshow('frame', frame_display)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
        except Exception as e:
            print(f"Error writing frame: {e}")

try:
    print("Video processing complete.")
    # Properly release resources
    out.release()
    cap.release()
    print(f"Output video saved to: {output_path}")
except Exception as e:
    print(f"Error during cleanup: {e}")
