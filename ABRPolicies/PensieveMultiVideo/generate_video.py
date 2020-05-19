import numpy as np

# Preparing Model for Twitch

RANDOM_SEED = 42
NUM_VIDEOS = 100
MAX_NUM_BITRATES = 10
MIN_NUM_BITRATES = 3
MAX_NUM_CHUNKS = 100
MIN_NUM_CHUNKS = 20
# bit rate candidates
# [200 1100 2000 2900 3800 4700 5600 6500 7400 8400]  # Kbps
MEAN_VIDEO_SIZE = [ 2.97, 11.13, 24.31, 32.87, 37.96, 48.97, 67.05, 67.05, 74.12,
       84.16]  # MB
STD_VIDEO_SIZE_NOISE = 0.1
VIDEO_FOLDER = './synthetic_videos/'


np.random.seed(RANDOM_SEED)
all_bitrate_idx = np.array(list(range(MAX_NUM_BITRATES)))
mask_bitrate_idx_to_shuffle = np.array(list(range(MAX_NUM_BITRATES)))

for video_idx in range(NUM_VIDEOS):
	num_bitrates = np.random.randint(MIN_NUM_BITRATES, MAX_NUM_BITRATES + 1)
	num_chunks = np.random.randint(MIN_NUM_CHUNKS, MAX_NUM_CHUNKS + 1)

	np.random.shuffle(mask_bitrate_idx_to_shuffle)

	mask_bitrate_idx = mask_bitrate_idx_to_shuffle[:num_bitrates]
	mask_bitrate_idx.sort()

	if np.all(mask_bitrate_idx == [1, 3, 4, 5, 6, 7]):
		# avoid using the same bitrates as the ones we do testing
		np.random.shuffle(mask_bitrate_idx_to_shuffle)
		mask_bitrate_idx = mask_bitrate_idx_to_shuffle[:num_bitrates]
		mask_bitrate_idx.sort()

	with open(VIDEO_FOLDER + str(video_idx), 'w') as f:
		f.write(str(num_bitrates) + '\t' + str(num_chunks) + '\n')
		for i in range(MAX_NUM_BITRATES):
			if i in mask_bitrate_idx:
				f.write('1' + '\t')
			else:
				f.write('0' + '\t')
		f.write('\n')

		for _ in range(num_chunks):
			for i in range(num_bitrates):
				mean = MEAN_VIDEO_SIZE[mask_bitrate_idx[i]]
				noise = np.random.normal(1, STD_VIDEO_SIZE_NOISE)
				f.write(str(mean * noise) + '\t')
			f.write('\n')	
