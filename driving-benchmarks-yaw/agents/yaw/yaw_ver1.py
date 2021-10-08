import os
import scipy
import scipy.misc

import torch
import numpy as np
import time

# from carla.agent import Agent
# from carla.carla_server_pb2 import Control

import agents.yaw.modules.yaw_net_v1 as resnet_carla

from version084.benchmark_tools.agent import Agent
from version084.carla.carla_server_pb2 import Control

from utils.max_branch_ver0_adj import action_adjusting, action_adjusting_town02

class ImitationLearning(Agent):

    def __init__(self, city_name,
                 avoid_stopping=True,
                 model_path="model/policy.pth",
                 visualize=False,
                 log_name="test_log",
                 image_cut=[115, 510]):

        super(ImitationLearning, self).__init__()
        # Agent.__init__(self)

        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        dir_path = os.path.dirname(__file__)
        self._models_path = os.path.join(dir_path, model_path)
        self.model = resnet_carla.resnet34_carla()
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
        self.before_direction = 0
        self._kP = 0.8
        self._kI = 0.5
        self._kD = 0.0
        self.i_term_previous = 0

    def load_model(self):
        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path: %s'
                               % self._models_path)
        checkpoint = torch.load(self._models_path, map_location='cuda:0')
        # checkpoint = torch.load(self._models_path)
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
            measurements, episode_name,
            directions)

        return control

    def _compute_action(self, rgb_image, measurements, episode_name, direction=None):

        speed = measurements.player_measurements.forward_speed
        loc_x = measurements.player_measurements.transform.location.x
        loc_y = measurements.player_measurements.transform.location.y
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

        if self.before_direction == 1 and direction == 2:
            direction = 0
        elif self.before_direction == 2 and direction == 1:
            direction = 0
        else:
            self.before_direction = direction


        # ImitationLearning.beforeImg = image_input

        steer, acc, brake, pred_speed = self._control_function(image_input, speed / 40, direction)

        self.running_cnt += 1

        # print('[[[[[[[[[[[ ', loc_x, loc_y, ' ]]]]]]]]]]]]]]]')

        ori_str = steer
        ori_brake = brake
        ori_acc = acc
        steer, acc, brake = action_adjusting(direction, steer, acc, brake, speed, pred_speed, episode_name, loc_x, loc_y, self.running_cnt)
        # steer, acc, brake = action_adjusting_town02(direction, steer, acc, brake, speed, pred_speed, episode_name, loc_x, loc_y, self.running_cnt)

        print('**[%d, %d] direc: %d, steer: %.3f (%.3f), break: %.3f (%.3f), acc: %.3f (%.3f), real speed: %.3f, pred_speed: %.3f '
              % (loc_x, loc_y, direction, steer, ori_str, brake, ori_brake, acc, ori_acc, speed, pred_speed))

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
            # branches, pred_speed = self.model(img_ts, speed_ts)
            pred_speed, pred_posi_x, pred_posi_y, pred_steering = self.model(img_ts, speed_ts)

        pred_posi_x = pred_posi_x[0][control_input].cpu().numpy()
        pred_posi_y = pred_posi_y[0][control_input].cpu().numpy()
        pred_steering = pred_steering[0][control_input].cpu().numpy()
        predicted_speed = pred_speed.squeeze().item()

        pred_steering += 0.01

        print('[[[ ', pred_posi_x, ' ', pred_posi_y, ' ]]] ' )

        real_speed = speed * 40.0
        real_predicted_speed = predicted_speed * 40.0

        predicted_acc, predicted_brake = self._gene_longi_control(real_speed, real_predicted_speed)

        # if self._avoid_stopping:
        if False:

            # print('real_speed: ', real_speed, ' real_predicted :', real_predicted)
            if real_speed < 1.0 and real_predicted_speed > 3.0:
                predicted_acc = 1 * (5.6 / 10.0 - speed) + predicted_acc
                predicted_brake = 0.0
                predicted_acc = predicted_acc

        return pred_steering, predicted_acc, predicted_brake, real_predicted_speed

    def _gene_longi_control(self, cur_speed, target_speed):
        time_step = 0.1
        speed_error = target_speed - cur_speed
        k_term = self._kP * speed_error
        i_term = self.i_term_previous + self._kI * time_step * speed_error
        self.i_term_previous = i_term

        cal_throttle = k_term + i_term
        cal_throttle = np.fmax(np.fmin(cal_throttle, 1.0), 0.0)

        if target_speed < 6 or speed_error < -3:
            cal_brake = 1
            cal_throttle = 0
        else:
            cal_brake = 0

        return cal_throttle, cal_brake
