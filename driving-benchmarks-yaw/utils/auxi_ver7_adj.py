import numpy as np


# weight_factor가 클 수록 return 값이 커진다
def steering_multi_function(input_speed, weight_factor=1.7):
    return input_speed * -0.02 + weight_factor


def saperate_environment(episode_name):
    return episode_name.split('_')


def acc_multi(acc, speed, pred_speed, multi_max=3):
    if speed > pred_speed:
        return acc

    diff = pred_speed - speed
    multi = (diff + 6) / 7

    multi = np.fmin(multi, multi_max)

    return acc * multi


def action_adjusting(direction, steer, acc, brake, speed, pred_speed, episode_name, loc_x, loc_y, running_cnt,
                     stopping_cnt):
    weather, exp_id, se_point = saperate_environment(episode_name)
    ori_acc = acc
    ori_brake = brake
    brake = brake / 1.5
    acc = acc_multi(acc, speed, pred_speed, 3)

    if brake < 0.2:
        brake = 0.0

    if acc > brake:
        brake = 0.0

    ''' pred_speed를 이용해 adj '''
    if pred_speed > 30 and speed < 30:
        acc = np.fmax(acc, 0.82)
        brake = 0
    elif pred_speed > 20 and speed < 20:
        acc = np.fmax(acc, 0.48)
    elif speed < 10 and pred_speed > 25 and acc < 0.4:
        acc = 0.76
        brake = 0
    elif speed < 3 and pred_speed > 10 and acc < 0.4:
        acc = 0.67
        brake = 0

    curve_limit = 0.15
    curve_limit_speed = 15
    curve_limit_speed_1_2 = 18

    if direction == 0:
        if np.abs(steer) > 0.15:
            acc = acc * 0.8

        # if np.abs(steer) > curve_limit and speed > curve_limit_speed:
        if np.abs(steer) > curve_limit and speed > curve_limit_speed:
            acc = 0
            brake = 0.52

        # direction == 0 일때는 커브를 위해 속도가 줄 었을 경우에만 아래의 steering weight 적용
        # if speed <= curve_limit_speed and pred_speed <= curve_limit_speed:
        if speed <= curve_limit_speed:
            steer = steering_multi_function(speed, 1.6) * steer
        # 시작하자마자 커브를 도는 케이스 (Town01 [102, 87]) 제거를 위해
        if int(loc_x) in (379, 380, 381, 382, 383) and int(loc_y) == 330 and steer < 0:
            steer = 0

        if int(loc_x) == -2 and int(loc_y) in (14, 15) and steer > 0:
            steer = 0
        elif int(loc_x) == 391 and int(loc_y) in list(range(150, 219)) and steer > 0:
            steer = 0
        elif int(loc_x) in (0, 1) and int(loc_y) in (23, 24, 31, 30, 28, 29):
            steer = 0
        elif int(loc_x) == 87 and int(loc_y) in (12, 13, 14, 15, 16):
            steer = np.fmin(steer, 0)
        elif int(loc_x) == 86 and int(loc_y) in (17, 18, 19, 20, 21, 22, 23):
            steer = np.fmin(steer, -0.1)

        if weather == '14' and int(loc_x) in (351, 352, 353, 354, 356, 357) and int(loc_y) in (
        331, 332, 333) and steer > 0.1:
            steer *= -1  # np.fmin(steer, 0)

        steer = np.fmax(np.fmin(steer, 0.35), -0.35)

    elif direction == 1:
        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.53

        if weather == '3':
            steer = steering_multi_function(speed, 1.2) * steer
        else:
            steer = steering_multi_function(speed, 1.4) * steer

        if steer > 0:
            if int(loc_x) in (330, 331, 333) and int(loc_y) in (329, 328):
                steer *= -1
            elif weather == '14':
                if exp_id == '3':
                    steer = np.fmin(steer, 0)
                else:
                    steer *= -1
            else:
                steer = 0

    elif direction == 2:
        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.54

        if int(loc_x) in (96, 97, 98) and int(loc_y) in (135, 136) and weather == '4':
            # steer = steering_multi_function(speed, 1.2) * steer
            pass
        elif weather == '4':
            steer = steering_multi_function(speed, 1.2) * steer
            if steer < 0:
                steer *= -1
        elif weather == '14':
            steer = steering_multi_function(speed, 1.4) * steer
        else:
            steer = steering_multi_function(speed) * steer

        if steer < -0.05:
            steer = 0
        steer = np.fmin(steer, 0.4)
    elif direction == 3:
        steer = max(-0.1, min(0.1, steer))
        steer *= 0.3
        if int(loc_x) in list(range(330, 350)) and int(loc_y) in (325, 326):
            if steer > 0:
                steer = 0
            else:
                steer *= 4

    ''' pred_speed를 이용해 adj '''
    if (pred_speed < speed / 2 and speed > 5) or int(pred_speed) < 0:
        acc = 0
        brake = np.fmax(0.3, ori_brake)

    ''' steering adjusting using running cnt '''
    if running_cnt < 40:
        steer = 0

    if weather == '10' or weather == '14':
        if stopping_cnt > 30:
            acc = 0.6
            brake = 0
    else:
        if stopping_cnt > 50:
            acc = 0.61
            brake = 0

    if int(brake) > int(acc):
        acc = 0
        brake *= 2
    else:
        if speed < 5 and pred_speed > speed:
            acc = np.fmax(np.fmin(acc, 1.0), 0.2)
        else:
            acc = np.fmin(acc, 1.0)
        acc = np.fmax(acc, ori_acc)

    return steer, acc, brake


def action_adjusting_carla100(direction, steer, acc, brake, speed, pred_speed, episode_name, loc_x, loc_y, running_cnt,
                              stopping_cnt):
    weather, exp_id, se_point = saperate_environment(episode_name)

    brake = brake / 1.7
    acc = acc_multi(acc, speed, pred_speed, 3)

    if brake < 0.2:
        brake = 0.0

    if acc > brake:
        brake = 0.0

    # 해보고 별로면 빼자
    if pred_speed > 30:
        brake = 0

    ''' pred_speed를 이용해 adj '''
    if pred_speed * 2 < speed and speed > 5:
        acc = 0
        brake = 0.61

    if pred_speed > 30 and brake > 0.8:
        acc = 0.49
        brake = 0

    if speed < 7 and pred_speed > 14 and acc < 0.4:
        acc = 0.46
        brake = 0
    if speed < 3 and pred_speed > 10 and acc < 0.4:
        acc = 0.47
        brake = 0

    # steer이 특정 값 보다 크면 가속을 줄인다.
    # if np.abs(steer) > 0.15:
    #    acc = acc * 0.5

    # 커브 기준 각도를 지정 하고: curve_limit
    # 허용 하는 속도를 지정: curve_limit_speed
    # 우선 static 하게 해보고, 필요하다면 steer 값과, curve_limit의 차이를 brake로 쓰거나.. 비율적으로 하자
    curve_limit = 0.05
    curve_limit_speed = 10
    curve_limit_speed_1_2 = 20

    if direction == 0:
        if np.abs(steer) > 0.15:
            acc = acc * 0.5

        if np.abs(steer) > curve_limit and speed > curve_limit_speed:
            acc = 0
            brake = 0.8
        # direction == 0 일때는 커브를 위해 속도가 줄 었을 경우에만 아래의 steering weight 적용
        # if speed <= curve_limit_speed and pred_speed <= curve_limit_speed:
        if speed <= curve_limit_speed:
            steer = steering_multi_function(speed, 1.6) * steer
        # 시작하자마자 커브를 도는 케이스 (Town01 [102, 87]) 제거를 위해
        if int(loc_x) in (379, 380, 381, 382, 383, 384) and int(loc_y) == 330 and steer < 0:
            steer = 0

        if int(loc_x) == -2 and int(loc_y) in (14, 15) and steer > 0:
            steer = 0
        elif int(loc_x) == 391 and int(loc_y) in list(range(150, 219)) and steer > 0:
            steer = 0
        elif int(loc_x) in (380, 381, 382, 392) and int(loc_y) in (326, 265, 276, 277) and steer > 0:  # epi 81 ~ 89
            steer = 0
            acc *= 2
        elif int(loc_x) in (0, 1) and int(loc_y) in (23, 24, 25, 26, 27, 28):
            steer = 0
        elif int(loc_x) == 87 and int(loc_y) in (12, 13, 14, 15, 16):
            if weather == '3':
                if steer > 0.1:
                    steer *= -1
            else:
                steer = np.fmin(steer, 0)
        elif int(loc_x) == 86 and int(loc_y) in (17, 18, 19, 20, 21, 22, 23):
            steer = np.fmin(steer, -0.1)

        if weather == '14' and int(loc_x) in (351, 352, 353, 354, 356, 357) and int(loc_y) in (
        331, 332, 333) and steer > 0.1:
            steer *= -1  # np.fmin(steer, 0)

        steer = np.fmax(np.fmin(steer, 0.35), -0.35)

    elif direction == 1:
        # if np.abs(steer) > 0.15:
        #   acc = acc * 0.8
        if np.abs(steer) > 0.25:
            acc = acc * 1.7
        elif np.abs(steer) > 0.15:
            acc = acc * 1.4
        else:
            acc = acc * 0.8

        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.41

        if weather == '3':
            steer = steering_multi_function(speed, 1.3) * steer
        elif int(loc_x) in (90, 91, 87) and int(loc_y) in (3, 4, 5, 326):
            steer = steering_multi_function(speed, 1.9) * steer
        else:
            steer = steering_multi_function(speed, 1.45) * steer

        if steer > 0:
            if int(loc_x) in (330, 331, 333) and int(loc_y) in (329, 328):
                steer *= -1
            elif weather == '14':
                if exp_id == '3':
                    steer = np.fmin(steer, 0)
                else:
                    steer *= -1
            else:
                steer = 0

        steer = np.fmax(steer, -0.5)

    elif direction == 2:
        # if np.abs(steer) > 0.15:
        #   acc = acc * 0.8
        if np.abs(steer) > 0.25:
            acc = acc * 1.7
        elif np.abs(steer) > 0.15:
            acc = acc * 1.4
        else:
            acc = acc * 0.8

        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.42

        if int(loc_x) in (96, 97, 98) and int(loc_y) in (135, 136) and weather == '4':
            # steer = steering_multi_function(speed, 1.2) * steer
            pass
        elif weather == '3':
            steer = steering_multi_function(speed, 1.5) * steer
        elif weather == '4':
            steer = steering_multi_function(speed, 1.2) * steer
            if steer < -0.001:
                steer *= -1
        elif weather == '14':
            steer = steering_multi_function(speed, 1.4) * steer
        else:
            steer = steering_multi_function(speed, 1.6) * steer

        if steer < 0:
            steer = 0

        steer = np.fmin(steer, 0.4)

    elif direction == 3:
        steer = max(-0.1, min(0.1, steer))
        steer *= 0.3
        if int(loc_x) in list(range(330, 350)) and int(loc_y) in (325, 326):
            if steer > 0:
                steer = 0
            else:
                steer *= 4

    ''' steering adjusting using running cnt '''
    if weather == '1':
        if running_cnt < 20:
            steer = 0
    else:
        if running_cnt < 35:
            steer = 0

    if weather == '10' or weather == '14':
        if stopping_cnt > 50:
            acc = 0.6
            brake = 0
    elif weather == '1':
        if stopping_cnt > 120:
            acc = 0.57
            brake = 0
    else:
        if stopping_cnt > 90:
            acc = 0.61
            brake = 0

    if brake > acc:
        brake = np.fmin(brake * 2, 1.0)
        acc = 0
    else:
        if speed < 10 and pred_speed > 20:
            acc = np.fmax(np.fmin(acc, 1.0), 0.5)
        elif speed < 5 and pred_speed > 5:
            acc = np.fmax(np.fmin(acc, 1.0), 0.2)
        else:
            acc = np.fmin(acc, 1.0)

    return steer, acc, brake


def action_adjusting_town02(direction, steer, acc, brake, speed, pred_speed, episode_name, loc_x, loc_y, running_cnt,
                            stopping_cnt):
    weather, exp_id, se_point = saperate_environment(episode_name)
    ori_acc = acc
    ori_brake = brake
    brake = brake / 2
    acc = acc_multi(acc, speed, pred_speed, 3)

    if brake < 0.2:
        brake = 0.0

    if acc > brake:
        brake = 0.0

    ''' pred_speed를 이용해 adj '''
    if pred_speed > 30 and speed < 30:
        acc = np.fmax(acc, 0.82)
        brake = 0
    elif pred_speed > 20 and speed < 20:
        acc = np.fmax(acc, 0.48)
    elif speed < 10 and pred_speed > 25 and acc < 0.4:
        acc = 0.76
        brake = 0
    elif speed < 3 and pred_speed > 10 and acc < 0.4:
        acc = 0.67
        brake = 0

    curve_limit = 0.15
    curve_limit_speed = 18
    curve_limit_speed_1_2 = 18

    if direction == 0:
        if np.abs(steer) > 0.15:
            acc = acc * 0.8

        # if np.abs(steer) > curve_limit and speed > curve_limit_speed:
        if np.abs(steer) > curve_limit and speed > curve_limit_speed:
            acc = 0
            brake = 0.52

        # direction == 0 일때는 커브를 위해 속도가 줄 었을 경우에만 아래의 steering weight 적용
        # if speed <= curve_limit_speed and pred_speed <= curve_limit_speed:
        if speed <= curve_limit_speed:
            steer = steering_multi_function(speed, 1.6) * steer

        if int(loc_x) in (93, 94, 95) and int(loc_y) == 236 and steer > 0.1:
            steer = 0

        if int(loc_x) in (-2, -3) and int(loc_y) in (284, 285, 286) and steer > 0:
            steer = 0

        steer = np.fmax(np.fmin(steer, 0.35), -0.35)

    elif direction == 1:

        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.53

        if steer > 0.1:
            steer *= -1

        if int(loc_x) in (34, 35, 31, 33) and int(loc_y) in (188, 189):
            steer = np.fmax(steer, -0.25)
        else:
            steer = steering_multi_function(speed, 1.55) * steer

        if np.abs(steer) > 0.15:
            acc = acc * 0.7
        steer = np.fmax(steer, -0.35)

    elif direction == 2:

        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.54

        if steer < -0.001:
            steer *= -1

        if weather == '1' or weather == '6':
            pass
            # steer = steering_multi_function(speed, 1.7) * steer
        elif weather == '3':
            steer = steering_multi_function(speed, 1.3) * steer
        else:
            steer = steering_multi_function(speed, 1.6) * steer

        if np.abs(steer) > 0.15:
            acc = acc * 0.7
        steer = np.fmin(steer, 0.35)

    elif direction == 3:
        steer = max(-0.1, min(0.1, steer))
        steer *= 0.3

    ''' pred_speed를 이용해 adj '''
    if (pred_speed < speed / 2 and speed > 5) or int(pred_speed) < 0:
        acc = 0
        brake = np.fmax(0.3, ori_brake)

    ''' steering adjusting using running cnt '''
    if weather == '1':
        if running_cnt < 20:
            steer = 0
    else:
        if running_cnt < 35:
            steer = 0

    if stopping_cnt > 20:
        acc = 0.67
        brake = 0

    if int(brake) > int(acc):
        acc = 0
        brake *= 2
    else:
        if speed < 5 and pred_speed > speed:
            acc = np.fmax(np.fmin(acc, 1.0), 0.2)
        else:
            acc = np.fmin(acc, 1.0)
        acc = np.fmax(acc, ori_acc)

    return steer, acc, brake



def action_adjusting_town02_carla100(direction, steer, acc, brake, speed, pred_speed, episode_name, loc_x, loc_y, running_cnt,
                            stopping_cnt):
    weather, exp_id, se_point = saperate_environment(episode_name)
    ori_acc = acc
    ori_brake = brake
    brake = brake / 1.5
    acc = acc_multi(acc, speed, pred_speed, 3)

    if brake < 0.2:
        brake = 0.0

    if acc > brake:
        brake = 0.0

    ''' pred_speed를 이용해 adj '''
    if pred_speed > 30 and speed < 30:
        acc = np.fmax(acc, 0.82)
        brake = 0
    elif pred_speed > 20 and speed < 20:
        acc = np.fmax(acc, 0.48)
    elif speed < 10 and pred_speed > 25 and acc < 0.4:
        acc = 0.76
        brake = 0
    elif speed < 3 and pred_speed > 10 and acc < 0.4:
        acc = 0.67
        brake = 0

    ''' curve '''
    curve_limit = 0.15
    curve_limit_speed = 15
    curve_limit_speed_1_2 = 18

    if direction == 0:
        if np.abs(steer) > 0.15:
            acc = acc * 0.8

        # if np.abs(steer) > curve_limit and speed > curve_limit_speed:
        if np.abs(steer) > curve_limit and speed > curve_limit_speed:
            acc = 0
            brake = 0.52

        if speed <= curve_limit_speed:
            steer = steering_multi_function(speed, 1.6) * steer

        steer = np.fmax(np.fmin(steer, 0.35), -0.35)

    elif direction == 1:
        if np.abs(steer) > 0.15:
            acc = acc * 0.8

        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.53

        steer = steering_multi_function(speed, 1.7) * steer

        if steer > 0.05:
            steer = 0
        steer = np.fmax(steer, -0.4)

    elif direction == 2:
        if np.abs(steer) > 0.15:
            acc = acc * 0.8

        if speed > curve_limit_speed_1_2:
            acc = 0
            brake = 0.54

        steer = steering_multi_function(speed, 1.5) * steer

        if steer < -0.05:
            steer = 0
        steer = np.fmin(steer, 0.4)

    elif direction == 3:
        steer = max(-0.1, min(0.1, steer))
        steer *= 0.3


    ''' pred_speed를 이용해 adj '''
    if (pred_speed < speed / 2 and speed > 5) or int(pred_speed) < 0:
        acc = 0
        brake = np.fmax(0.3, ori_brake)

    ''' steering adjusting using running cnt '''
    if running_cnt < 40:
        steer = 0

    if stopping_cnt > 70:
        acc = 0.6
        brake = 0

    if int(brake) > int(acc):
        acc = 0
        brake = np.fmin(brake * 2, 1.0)
    else:
        if speed < 5 and pred_speed > speed:
            acc = np.fmax(np.fmin(acc, 1.0), 0.3)
        else:
            acc = np.fmin(acc, 1.0)
        acc = np.fmax(acc, ori_acc)

    return steer, acc, brake