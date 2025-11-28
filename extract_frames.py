import cv2
import os


def extract_frames(video_path, output_folder, video_index, frame_interval=6):
    """
    从视频中抽取帧并保存到本地文件夹

    :param video_path: 输入视频路径
    :param output_folder: 保存帧的文件夹
    :param frame_interval: 每隔多少帧抽取一帧
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件！")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取完毕

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{video_index}_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"{video_index}.mp4,共抽取 {saved_count} 帧。")


if __name__ == "__main__":
    root = r"G:\dataset\ucf_crimes\video"
    save_root = r"G:\dataset\ucf_crimes\data"
    list_videos = os.listdir(root)

    for video_name in list_videos:
        video_index = video_name.split(".")[0]
        video_path = os.path.join(root,video_name)
        output_folder = os.path.join(save_root,str(video_index))
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        extract_frames(video_path,output_folder,video_index)