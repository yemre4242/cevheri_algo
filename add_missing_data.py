import csv
import numpy as np
from scipy.interpolate import interp1d
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


def interpolate_gun_bboxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    gun_ids = np.array([int(float(row['gun_id'])) for row in data])
    gun_bboxes = np.array([eval(row['bbox']) for row in data])
    scores = np.array([float(row['score']) for row in data])
    class_ids = np.array([int(float(row['class_id'])) for row in data])

    interpolated_data = []
    unique_gun_ids = np.unique(gun_ids)
    for gun_id in unique_gun_ids:
        frame_numbers_ = [int(p['frame_nmr']) for p in data if int(float(p['gun_id'])) == int(float(gun_id))]
        gun_mask = gun_ids == gun_id
        gun_frame_numbers = frame_numbers[gun_mask]
        gun_bboxes_interpolated = []
        scores_interpolated = []
        class_ids_interpolated = []

        first_frame_number = gun_frame_numbers[0]
        last_frame_number = gun_frame_numbers[-1]

        for i in range(len(gun_bboxes[gun_mask])):
            frame_number = gun_frame_numbers[i]
            gun_bbox = gun_bboxes[gun_mask][i]
            score = scores[gun_mask][i]
            class_id = class_ids[gun_mask][i]

            if i > 0:
                prev_frame_number = gun_frame_numbers[i-1]
                prev_gun_bbox = gun_bboxes_interpolated[-1]
                prev_score = scores_interpolated[-1]
                prev_class_id = class_ids_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_gun_bbox, gun_bbox)), axis=0, kind='linear')
                    interpolated_gun_bboxes = interp_func(x_new)
                    # Skor ve class_id için lineer interpolasyon (veya sabit değer)
                    interpolated_scores = np.linspace(prev_score, score, num=frames_gap, endpoint=False)
                    interpolated_class_ids = np.linspace(prev_class_id, class_id, num=frames_gap, endpoint=False)

                    gun_bboxes_interpolated.extend(interpolated_gun_bboxes[1:])
                    scores_interpolated.extend(interpolated_scores[1:])
                    class_ids_interpolated.extend(interpolated_class_ids[1:])

            gun_bboxes_interpolated.append(gun_bbox)
            scores_interpolated.append(score)
            class_ids_interpolated.append(class_id)

        for i in range(len(gun_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['gun_id'] = str(gun_id)
            row['bbox'] = str(list(gun_bboxes_interpolated[i]))
            row['score'] = str(scores_interpolated[i])
            row['class_id'] = str(int(class_ids_interpolated[i]))
            if frame_number not in frame_numbers_:
                row['interpolated'] = '1'
            else:
                row['interpolated'] = '0'
            interpolated_data.append(row)
    return interpolated_data


# Load the CSV file
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

# --- Silah interpolasyonu ---
try:
    with open('guns.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    interpolated_data = interpolate_gun_bboxes(data)
    header = ['frame_nmr', 'gun_id', 'bbox', 'score', 'class_id', 'interpolated']
    with open('guns_interpolated.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)
    print('guns_interpolated.csv dosyası oluşturuldu.')
except Exception as e:
    print('Silah interpolasyonu sırasında hata:', e)
