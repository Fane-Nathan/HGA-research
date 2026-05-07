import random
import numpy as np

from tmrl.memory import TorchMemory
import tmrl.config.config_constants as cfg

# LOCAL BUFFER COMPRESSION ==============================


def get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition (but not influencing the unaugmented observation in real-time envs)
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1][-19:])  # speed and most recent LIDAR only
    rew_mod = np.float32(rew)
    terminated_mod = terminated
    truncated_mod = truncated
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info


def get_local_buffer_sample_lidar_progress(prev_act, obs, rew, terminated, truncated, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition (but not influencing the unaugmented observation in real-time envs)
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1], obs[2][-19:])  # speed and most recent LIDAR only
    rew_mod = np.float32(rew)
    terminated_mod = terminated
    truncated_mod = truncated
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info):
    """
    Sample compressor for MemoryTMFull
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    prev_act_mod = prev_act
    obs_mod = (obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8))
    rew_mod = rew
    terminated_mod = terminated
    truncated_mod = truncated
    info_mod = info
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod


def get_local_buffer_sample_tm20_hybrid(prev_act, obs, rew, terminated, truncated, info):
    """
    Sample compressor for MemoryTMHybrid (CNN + Lidar + XYZ + Progress)
    Input:
        prev_act: action computed from a previous observation
        obs: (speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain) after preprocessing
        rew, terminated, truncated, info: outcome of the transition
    Compressed format: (speed, gear, rpm, image_single, lidar, xyz, progress, crash, progress_gain)
    CAUTION: prev_act is the action that comes BEFORE obs
    """
    prev_act_mod = prev_act
    # obs[3] = images (4 frames), we take only the last frame and convert to uint8
    crash = obs[7] if len(obs) > 7 else np.array([0.0], dtype=np.float32)
    progress_gain = obs[8] if len(obs) > 8 else np.array([0.0], dtype=np.float32)
    obs_mod = (
        obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8),
        obs[4], obs[5], obs[6], crash, progress_gain
    )
    rew_mod = rew
    terminated_mod = terminated
    truncated_mod = truncated
    info_mod = info
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod


# FUNCTIONS ====================================================


def last_true_in_list(li):
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_eoe(hist, eoe_idx_in_hist):
    """
    Pads the history hist before the End Of Episode (EOE) index.

    Previous entries in hist are padded with copies of the first element occurring after EOE.
    """
    last_idx = len(hist) - 1
    assert eoe_idx_in_hist <= last_idx, f"replace_hist_before_eoe: eoe_idx_in_hist:{eoe_idx_in_hist}, last_idx:{last_idx}"
    if 0 <= eoe_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):
            if i <= eoe_idx_in_hist:
                hist[i] = hist[i + 1]


# SUPPORTED CUSTOM MEMORIES ============================================================================================


class GenericTorchMemory(TorchMemory):
    def __init__(self,
                 memory_size=1e6,
                 batch_size=1,
                 dataset_path="",
                 nb_steps=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):

        # parse:
        d0 = [b[0] for b in buffer.memory]  # actions
        d1 = [b[1] for b in buffer.memory]  # observations
        d2 = [b[2] for b in buffer.memory]  # rewards
        d3 = [b[3] for b in buffer.memory]  # terminated
        d4 = [b[4] for b in buffer.memory]  # truncated
        d5 = [b[5] for b in buffer.memory]  # info
        d6 = [b[3] or b[4] for b in buffer.memory]  # done

        # append:
        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        # trim
        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):

        # This is a hack to avoid invalid transitions from terminal to initial
        # TODO: find a way to only index valid transitions instead
        while self.data[6][item]:
            item = random.randint(a=0, b=self.__len__() - 1)

        idx_last = item
        idx_now = item + 1

        last_obs = self.data[1][idx_last]
        new_act = self.data[0][idx_now]
        rew = self.data[2][idx_now]
        new_obs = self.data[1][idx_now]
        terminated = self.data[3][idx_now]
        truncated = self.data[4][idx_now]
        info = self.data[5][idx_now]

        return last_obs, new_act, rew, new_obs, terminated, truncated, info


class MemoryTM(TorchMemory):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):
        raise NotImplementedError

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        raise NotImplementedError


class MemoryTMLidar(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[7][idx_now]
        truncated = self.data[8][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes (terminated or truncated)
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[3] for b in buffer.memory]  # terminated
        d8 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]

        return self


class MemoryTMLidarProgress(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[8][idx_now]
        truncated = self.data[9][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, truncated, terminated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][2] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # progress
        d8 = [b[3] for b in buffer.memory]  # terminated
        d9 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]

        return self


class MemoryTMFull(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], self.data[8][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], self.data[8][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[9][idx_now]
        truncated = self.data[10][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][3] for b in buffer.memory]  # images
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # gears
        d8 = [b[1][2] for b in buffer.memory]  # rpms
        d9 = [b[3] for b in buffer.memory]  # terminated
        d10 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]

        return self


class MemoryTMHybrid(MemoryTM):
    """
    Memory class for TM2020 Hybrid (CNN + Lidar) observations.
    Observation format:
    (speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2)
    """
    def __init__(self, *args, **kwargs):
        self.K_context = 16  # number of sequence steps for context
        super().__init__(*args, **kwargs)

    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:
                item += 1
            elif item == self.__len__() - 1:
                item -= 1
            elif random.random() < 0.5:
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # Handle EOE (end-of-episode) transitions
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]
        last_eoe_idx = last_true_in_list(last_eoes)

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        # Safe fallback for legacy replay buffers lacking privileged hybrid state.
        # np.atleast_1d ensures shape is always (N,) even if the stored value is a bare scalar
        xyz_last = np.atleast_1d(self.data[12][idx_last]).astype(np.float32) if len(self.data) > 12 else np.zeros(3, dtype=np.float32)
        prog_last = np.atleast_1d(self.data[13][idx_last]).astype(np.float32) if len(self.data) > 13 else np.zeros(1, dtype=np.float32)
        crash_last = np.atleast_1d(self.data[14][idx_last]).astype(np.float32) if len(self.data) > 14 else np.zeros(1, dtype=np.float32)
        gain_last = np.atleast_1d(self.data[15][idx_last]).astype(np.float32) if len(self.data) > 15 else np.zeros(1, dtype=np.float32)
        xyz_now = np.atleast_1d(self.data[12][idx_now]).astype(np.float32) if len(self.data) > 12 else np.zeros(3, dtype=np.float32)
        prog_now = np.atleast_1d(self.data[13][idx_now]).astype(np.float32) if len(self.data) > 13 else np.zeros(1, dtype=np.float32)
        crash_now = np.atleast_1d(self.data[14][idx_now]).astype(np.float32) if len(self.data) > 14 else np.zeros(1, dtype=np.float32)
        gain_now = np.atleast_1d(self.data[15][idx_now]).astype(np.float32) if len(self.data) > 15 else np.zeros(1, dtype=np.float32)

        # Observation format: (speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2)
        last_obs = (self.data[2][idx_last], self.data[7][idx_last], self.data[8][idx_last], 
                    imgs_last_obs, self.data[11][idx_last], xyz_last, prog_last, crash_last, gain_last, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], self.data[8][idx_now], 
                   imgs_new_obs, self.data[11][idx_now], xyz_now, prog_now, crash_now, gain_now, *new_act_buf)
        terminated = self.data[9][idx_now]
        truncated = self.data[10][idx_now]
        info = self.data[6][idx_now]

        # === Build context window for Context Encoder ===
        # Build K+1 steps so trainer can use:
        # - ctx      = context[:-1]  (ends at t)
        # - ctx_next = context[1:]   (ends at t+1)
        # Telemetry Context: speed(1) + lidar(19) + action(3) + reward(1) = 24
        K = self.K_context
        context = np.zeros((K + 1, 24), dtype=np.float32)
        # Image Context: 1 frame per step, size 64x64
        img_context = np.zeros((K + 1, 1, cfg.IMG_HEIGHT, cfg.IMG_WIDTH), dtype=np.float32)

        total_data_len = len(self.data[0])
        for k in range(K + 1):
            ctx_idx = idx_now - K + k
            if ctx_idx < 0 or ctx_idx >= total_data_len:
                continue  # leave as zeros (padding)

            # If this is an EOE from a previous episode token, wipe prefix.
            # Keep k == K intact so target (t+1) can still use its own terminal context.
            if self.data[4][ctx_idx] and k < K:
                context[:k + 1] = 0.0
                img_context[:k + 1] = 0.0
                continue

            spd = np.float32(self.data[2][ctx_idx]).reshape(-1)[-1]
            lid = np.array(self.data[11][ctx_idx], dtype=np.float32).flatten()
            act_ctx = np.array(self.data[1][ctx_idx], dtype=np.float32).flatten()
            rew_ctx = np.atleast_1d(self.data[5][ctx_idx]).astype(np.float32).reshape(-1)[-1]
            context[k, 0] = spd
            context[k, 1:20] = lid
            context[k, 20:23] = act_ctx
            context[k, 23] = rew_ctx

            # Get image for this step
            img_frame = np.array(self.data[3][ctx_idx], dtype=np.float32) / 256.0
            img_context[k, 0] = img_frame

        # Note: Added img_context. To avoid breaking things that expect exactly 8 outputs,
        # we append it to the tuple.
        return last_obs, new_act, rew, new_obs, terminated, truncated, info, context, img_context

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, terminated, truncated, info)
        obs format from worker:
        (speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2)
        In the compressed sample, obs keeps the latest image frame plus numeric fields.
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][3] for b in buffer.memory]  # images (single frame, compressed)
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # gears
        d8 = [b[1][2] for b in buffer.memory]  # rpms
        d9 = [b[3] for b in buffer.memory]  # terminated
        d10 = [b[4] for b in buffer.memory]  # truncated
        d11 = [b[1][4] if len(b[1]) > 4 else np.zeros(19, dtype=np.float32) for b in buffer.memory]  # lidar
        d12 = [b[1][5] if len(b[1]) > 5 else np.zeros(3, dtype=np.float32) for b in buffer.memory]  # xyz
        d13 = [b[1][6] if len(b[1]) > 6 else np.array([0.0], dtype=np.float32) for b in buffer.memory]  # progress
        d14 = [b[1][7] if len(b[1]) > 7 else np.array([0.0], dtype=np.float32) for b in buffer.memory]  # crash
        d15 = [b[1][8] if len(b[1]) > 8 else np.array([0.0], dtype=np.float32) for b in buffer.memory]  # progress_gain

        # Check and pad if loaded from a checkpoint with missing hybrid arrays.
        if self.__len__() > 0:
            while len(self.data) < 16:
                pad_len = len(self.data[0])
                if len(self.data) == 12:
                    self.data.append([np.zeros(3, dtype=np.float32) for _ in range(pad_len)])  # 12: xyz
                elif len(self.data) == 13:
                    self.data.append([np.array([0.0], dtype=np.float32) for _ in range(pad_len)])  # 13: progress
                elif len(self.data) == 14:
                    self.data.append([np.array([0.0], dtype=np.float32) for _ in range(pad_len)])  # 14: crash
                elif len(self.data) == 15:
                    self.data.append([np.array([0.0], dtype=np.float32) for _ in range(pad_len)])  # 15: progress_gain
                else:
                    break

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
            self.data[11] += d11
            self.data[12] += d12
            self.data[13] += d13
            self.data[14] += d14
            self.data[15] += d15
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)
            self.data.append(d11)
            self.data.append(d12)
            self.data.append(d13)
            self.data.append(d14)
            self.data.append(d15)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]
            self.data[11] = self.data[11][to_trim:]
            self.data[12] = self.data[12][to_trim:]
            self.data[13] = self.data[13][to_trim:]
            self.data[14] = self.data[14][to_trim:]
            self.data[15] = self.data[15][to_trim:]

        return self
