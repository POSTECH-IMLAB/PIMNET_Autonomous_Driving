import os
import scipy
import scipy.misc

import torch
import numpy as np

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from agents.imitation.modules.carla_net import CarlaNet
import time


class ImitationLearning(Agent):

    def __init__(self, city_name,
                 avoid_stopping=True,
                 model_path="model/policy.pth",
                 visualize=False,
                 log_name="test_log",
                 gpu_num = 3,
                 image_cut=[115, 510]):

        super(ImitationLearning, self).__init__()
        # Agent.__init__(self)

        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        dir_path = os.path.dirname(__file__)
        self._models_path = os.path.join(dir_path, model_path)
        # by kimna <===== input image의 크기에 따라 iptImgCnt 값을 바꿔라
        self.gpu_num = gpu_num
        self.iptImgCnt = 1
        self.model = CarlaNet(self.iptImgCnt)
        if torch.cuda.is_available():
            self.model.cuda()
        self.load_model()
        self.model.eval()

        self._image_cut = image_cut

        # by kimna
        self.episode_init_flag = 0
        self.before_episode_name = ''
        self.before_img = []
        self.stopping_cnt = 0
        self.running_cnt = 0

    def load_model(self):
        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path: %s'
                               % self._models_path)
        # checkpoint = torch.load(self._models_path, map_location='cuda:0')
        checkpoint = torch.load(self._models_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def run_step(self, measurements, sensor_data, directions, target, episode_name):

        # by kimna
        # When Episode Init
        if self.before_episode_name == '':
            self.before_episode_name = episode_name
            self.episode_init_flag = 1
            self.stopping_cnt = 0
            self.running_cnt = 0
        elif not self.before_episode_name == episode_name:
            self.before_episode_name = episode_name
            self.episode_init_flag = 1
            self.stopping_cnt = 0
            self.running_cnt = 0

        start = time.time()
        control = self._compute_action(
            sensor_data['CameraRGB'].data,
            measurements.player_measurements.forward_speed,
            directions)
        print('end time: ', time.time() - start)

        return control

    def _compute_action(self, rgb_image, speed, direction=None):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.expand_dims(np.transpose(image_input, (2, 0, 1)), axis=0)

        image_input = np.multiply(image_input, 1.0 / 255.0)
        speed = np.array([[speed]]).astype(np.float32) * 3.6

        # speed = np.array([[speed]]).astype(np.float32) / 10 # by kimna
        direction = int(direction-2)
        if direction == -2:
            direction = 0

        # before frame ipt saving --- by kimna
        if self.iptImgCnt == 2:
            if self.episode_init_flag == 1: # when episode init
                self.before_img = image_input
                image_input = np.concatenate((image_input, image_input), axis=1)
                self.episode_init_flag = 0
            else:    # when episode not init
                tmp_image = image_input
                image_input = np.concatenate((image_input, self.before_img), axis=1)    # 순서 바꾸자 by kimna
                self.before_img = tmp_image

        # ImitationLearning.beforeImg = image_input

        steer, acc, brake = self._control_function(image_input, speed / 90, direction)

        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 35: # and brake == 0.0:
            acc = 0.0

        if np.abs(steer) > 0.15:
            acc = acc * 0.4

        ########################## by kimna Conditions Start ##########################
        # if direction == 3 and np.abs(steer) > 0.1:
        #     steer = steer * 0.5
        #     if steer > 0.1:
        #         steer = 0.1
        #     elif steer < -0.1:
        #         steer = -0.1
        #
        # if acc > brake * 0.5:
        #     brake = 0.0
        #
        # if brake - acc > 0.5 and self.stopping_cnt < 5:
        #     brake = 0.0
        #     self.stopping_cnt = self.stopping_cnt + 1
        #     self.running_cnt = 0
        #     acc = 0.5

        if self.running_cnt < 300:
            self.running_cnt = self.running_cnt + 1
        else:
            self.stopping_cnt = 0

        if direction == 1 or direction == 2:
            acc = acc * 1.5
            # steer = steer * 1.4
            # if speed > 15:
            #     acc = acc * 2.5
            steer = self._steering_multi_function(speed) * steer
            # steer = steer - abs(steer) * 0.1
        elif direction == 0:
            acc = acc * 1.6
            steer = self._steering_multi_function(speed) * steer
        elif direction == 3:
            steer = self._steering_multi_function(speed) * steer
            # steer = steer - abs(steer) * 0.1

        # print("Steering Multi: ", self._steering_multi_function(speed))

        # if direction == 1 or direction == 2:
        ########################## by kimna Conditions End ##########################

        print('**** direc: %d, steer: %f, break: %f, acc: %f, real speed: %f' %(direction, steer, brake, acc, speed))
        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input):

        img_ts = torch.from_numpy(image_input).cuda()
        speed_ts = torch.from_numpy(speed).cuda()

        with torch.no_grad():
            branches, pred_speed = self.model(img_ts, speed_ts)

        pred_result = branches[0][
            3*control_input:3*(control_input+1)].cpu().numpy()

        predicted_steers = (pred_result[0])

        predicted_acc = (pred_result[1])

        predicted_brake = (pred_result[2])

        if self._avoid_stopping:
        # if False:
            predicted_speed = pred_speed.squeeze().item()
            real_speed = speed * 90.0

            real_predicted = predicted_speed * 90.0

            print('real_speed: ', real_speed, ' real_predicted :', real_predicted)

            if real_speed < 1.0 and real_predicted > 3.0:

                predicted_acc = 1 * (5.6 / 10.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc

        return predicted_steers, predicted_acc, predicted_brake

    def _steering_multi_function(self, input_steering):
        return input_steering * -0.02 + 1.7