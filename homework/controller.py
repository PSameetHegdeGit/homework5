import math

import pystk
import numpy as np
import math

def angle_between_vectors(a, b):
    dot_product = np.dot(a, b)
    magnitudes = np.linalg.norm(a) * np.linalg.norm(b)
    cos_theta = dot_product / magnitudes
    theta = math.acos(cos_theta)
    return theta

def rotation(a, b):
    cross_product = np.cross(a, b)
    if cross_product > 0:
        return -1
    else:
        return 1


def control(aim_point, current_vel, target_velocity=35):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    brake = False
    nitro = False
    aim_point = np.array([aim_point[0], aim_point[1] * -1]) # multiply by -1 to be able to properly calculate angle
    straight_vector = np.array([0,1])

    # Steering
    angle_in_radians = angle_between_vectors(straight_vector, aim_point)
    direction = rotation(straight_vector, aim_point)

    steep_turn_threshold = 0.15 * np.pi
    drift_threshold = 0.35 * np.pi

    steer = np.clip(direction * 10 * (angle_in_radians / (np.pi)), a_min=-1, a_max=1)
    drift = False

    acceleration = np.clip(2 * (1 - (current_vel / target_velocity)), a_min=0.001, a_max=1)

    if steep_turn_threshold <= angle_in_radians <= drift_threshold:
        drift = False
        brake = True
        # as turn increases, acceleration should decrease proportionally
        acceleration = np.clip((1 - (current_vel / (target_velocity/2))), a_min=0.001, a_max=1)
    elif angle_in_radians >= drift_threshold:
        drift = True
        brake = True
        # as turn increases, acceleration should decrease proportionally
        acceleration = np.clip((1 - (current_vel / (target_velocity/8))), a_min=0.001, a_max=1)

    #nitro edge case
    if angle_in_radians < 0.08:
        nitro = True

    #consider an edge case if aim_point is behind us
    if angle_in_radians > 0.75 * np.pi:
        drift = False


    # print('angle_in_radians:', angle_in_radians)
    #print('acceleration:', acceleration)
   #print('current_vel:', current_vel)

    # Break
    if current_vel > target_velocity:
        brake = True

    action = pystk.Action(steer=steer, brake=brake, acceleration=acceleration, drift=drift, nitro=nitro)


    return action


if __name__ == '__main__':
    from homework.utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
