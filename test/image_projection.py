import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def cut_image(image) :
    width, height = image.size

    top_half = image.crop((0, 0, width, height // 2))
    bottom_half = image.crop((0, height // 2, width, height))

    return top_half, bottom_half

def concat_image(top, bottom):
    # 두 이미지의 너비와 높이 가져오기
    top_width, top_height = top.size
    bottom_width, bottom_height = bottom.size

    # 두 이미지의 너비가 다르면 오류 처리 (합치기 전에 맞춰야 함)
    if top_width != bottom_width:
        raise ValueError("두 이미지의 너비가 같아야 합칠 수 있습니다.")

    # 새로운 이미지 생성 (높이는 두 이미지의 높이 합, 너비는 동일)
    new_image = Image.new('RGB', (top_width, top_height + bottom_height))
    # 이미지를 위에서부터 순서대로 붙이기
    new_image.paste(top, (0, 0))  # 위쪽 이미지 붙이기
    new_image.paste(bottom, (0, top_height))  # 아래쪽 이미지 붙이기
    return new_image

def print_image(image) :
    plt.imshow(image)
    plt.show()

def warp_image_for_projection(image):
    # 이미지 크기 (너비, 높이)
    h, w = image.shape[:2]

    # 원본 이미지의 4개 코너 좌표 (좌상, 우상, 좌하, 우하)
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # 꺾인 벽과 천장에 맞는 타겟 좌표 (변형 후)
    # 위쪽 절반은 천장, 아래쪽 절반은 벽에 맞도록 조정
    dst_points = np.float32([[0, 0], [w, 50], [0, h], [w, h - 50]])

    # 변환 매트릭스 계산
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 이미지 변환 적용
    warped_image = cv2.warpPerspective(image, M, (w, h))

    return warped_image


root = '/home/gkalstn000/lg'
filenames = os.listdir(root)

for filename in filenames:
    if 'origin' not in filename: continue
    image = Image.open(os.path.join(root, filename))
    image_warped = warp_image_for_projection(np.array(image))
    image_warped = Image.fromarray(image_warped)
    image_warped.save(os.path.join(root, filename.replace('origin', 'warped')))
    # w, h = image.size
    #
    # top, bottom = cut_image(image)
    #
    # h_resized = int((h / 2**0.5) / 2)
    #
    # top_resized = top.resize((w, h_resized))
    # bot_resized = bottom.resize((w, h_resized))
    #
    # concated = concat_image(top, bot_resized)
    # concated.save(os.path.join(root, filename.replace('origin', 'distort')))