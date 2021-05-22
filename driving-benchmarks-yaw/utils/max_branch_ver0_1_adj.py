import numpy as np


def steering_multi_function(input_speed, weight_factor=1.7):
    return input_speed * -0.02 + weight_factor

def saperate_environment(episode_name):
    return episode_name.split('_')


def action_adjusting(direction, steer, acc, brake, speed, episode_name):

    weather, exp_id, se_point = saperate_environment(episode_name)

    if exp_id == "3":
        if brake < 0.1:
            brake = 0.0
        else:
            brake *= 5

        if acc > 0.8:
            brake = 0.0

    else:
        if brake < 0.2:
            brake = 0.0

        if acc > brake:
            brake = 0.0

    # We limit speed to 35 km/h to avoid
    # for Town 1
    # if speed > 35: # and brake == 0.0:
    #     acc = 0.0
    # for Town 2

    # if self.before_steering != 0 and abs(self.before_steering - steer) > 0.1:
    #         steer = (self.before_steering + steer) / 2

    if exp_id == "0":
        if speed > 37:
            acc = 0.0
        acc = acc * 3.7
    elif exp_id == "3":
        if speed > 35:
            acc = 0.0
        acc = acc * 3.2
    else:
        if speed > 35:
            acc = 0.0
        acc = acc * 3.5

    # steer이 특정 값 보다 크면 가속을 줄인다.
    if np.abs(steer) > 0.15:
        acc = acc * 0.5

    # 커브 기준 각도를 지정 하고: curve_limit
    # 허용 하는 속도를 지정: curve_limit_speed
    # 우선 static 하게 해보고, 필요하다면 steer 값과, curve_limit의 차이를 brake로 쓰거나.. 비율적으로 하자
    curve_limit = 0.05
    curve_limit_speed = 15
    curve_limit_1_2 = 0.01
    curve_limit_speed_1_2 = 17

    if direction == 0:
        if np.abs(steer) > curve_limit and speed > curve_limit_speed:
            acc = 0
            brake = 1
        # direction == 0 일때는 커브를 위해 속도가 줄 었을 경우에만 아래의 steering weight 적용
        if speed <= curve_limit_speed:
            steer = steering_multi_function(speed) * steer
    elif direction == 1:
        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 1
        steer = steering_multi_function(speed, 1.7) * steer
        if steer > 0:
            steer = 0
    elif direction == 2:
        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 1
        steer = steering_multi_function(speed) * steer
        if steer < 0:
            steer = 0

    return steer, acc, brake


def action_adjusting_town02(direction, steer, acc, brake, speed, episode_name):
    weather, exp_id, se_point = saperate_environment(episode_name)

    if brake < 0.1:
        brake = 0.0

    if acc > brake:
        brake = 0.0

    # We limit speed to 35 km/h to avoid
    # for Town 1
    # if speed > 35: # and brake == 0.0:
    #     acc = 0.0
    # for Town 2

    if exp_id == "0":
        if speed > 37:
            acc = 0.0
        acc = acc * 1.7
    else:
        if speed > 35:
            acc = 0.0
        acc = acc * 1.5

    # steer이 특정 값 보다 크면 가속을 줄인다.
    if np.abs(steer) > 0.15:
        acc = acc * 0.5

    curve_limit = 0.04
    curve_limit_speed = 15
    curve_limit_1_2 = 0.01
    curve_limit_speed_1_2 = 17

    if direction == 0:
        if np.abs(steer) > curve_limit and speed > curve_limit_speed:
            acc = 0
            brake = 1
        # direction == 0 일때는 커브를 위해 속도가 줄 었을 경우에만 아래의 steering weight 적용
        if speed <= curve_limit_speed:
            steer = steering_multi_function(speed) * steer
    elif direction == 1:

        if weather == "14":
            if speed > curve_limit_speed_1_2 - 1:
                acc = 0
                brake = 1
            steer = steering_multi_function(speed, 1.55) * steer
        else:
            if speed > curve_limit_speed_1_2:
                acc = 0
                brake = 1
            steer = steering_multi_function(speed, 1.5) * steer
        if steer > 0:
            steer = 0
    elif direction == 2:
        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 1
        steer = steering_multi_function(speed) * steer
        if steer < 0:
            steer = 0

    return steer, acc, brake