import numpy as np
from utils.visualization import stgcn_visualize
import os
import skvideo

# 用来验证npy数据是否正确，会生成一个骨架视频
if __name__ == '__main__':
    # load data
    data_path = "data/nmv/train_data.npy"
    index = 50
    data = np.load(data_path)

    # select a sample
    sample = data[index]
    print (sample.shape)

    # simulate parameters
    num_node = 18
    self_link = [(i, i) for i in range(num_node)]
    neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                11),
                        (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                        (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    edge = self_link + neighbor_link

    intensity = np.ones((3, 300, 18, 1))
    H = 256
    W = 340
    video = np.zeros((300,H,W,3))

    # render the video
    images = stgcn_visualize(sample, edge, intensity, video)

    # save video
    output_result_dir = "data/nmv"
    video_name = "skeleton_validation"
    output_result_path = '{}/{}_{}.mp4'.format(output_result_dir, video_name, index)
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
    writer = skvideo.io.FFmpegWriter(output_result_path,
                                        outputdict={'-b': '300000000'})
    for img in images:
        img = img.astype(np.uint8)[:,:,[2,1,0]]
        # print('img.shape: {}'.format(img.shape))
        writer.writeFrame(img)
    writer.close()
    print("done")