# %%
from tello import Tello

# %%


class TelloControl(object):
    def __init__(self, tello: Tello):
        self.tello = tello 

        # RC control velocities
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0

    def control(self, hand_recognized):
        self.tello.send_command('up 25')


# %% [markdown]
# tello.connect()
# tello.takeoff()
#
# tello.move_left(100)
# tello.rotate_clockwise(90)
# tello.move_forward(100)
#
# tello.land()

# %%
