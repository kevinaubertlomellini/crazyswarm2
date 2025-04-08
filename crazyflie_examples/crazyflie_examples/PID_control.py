"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import LogDataGeneric

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 2.5

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(actual_positions, desired_positions):
    actual_positions = np.array(actual_positions)[1:]  # Skip the first entry
    desired_positions = np.array(desired_positions)[1:]  # Skip the first entry
    
    desired_positions[0] = actual_positions[0]
    
    time_steps = np.arange(len(actual_positions)) * 0.03

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    labels = ['X Position', 'Y Position', 'Z Position']
    for i in range(3):
        axs[i].plot(time_steps, actual_positions[:, i], label="Measured", color='k')
        axs[i].plot(time_steps, desired_positions[:, i], label="Desired", color='r')
        axs[i].set_ylabel('Distance (m)')
        axs[i].set_title(f"{labels[i]} Position")  # Adding individual subplot titles
        axs[i].legend()
        axs[i].grid(True)
        
    axs[1].set_ylim(-0.25, 0.25)  # Set y-axis range for the Y position subplot
    axs[2].set_ylim( 0.75, 1.25)  # Set y-axis range for the Y position subplot
    axs[2].set_xlabel("Time (s)")
    plt.suptitle("Crazyflie Position Tracking")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    plt.show()



class Position_Subscriber(Node):
    def __init__(self, drone_id, drone_controller):
        super().__init__('pos_subscriber_{}'.format(drone_id))
        self.drone_id = drone_id
        self.drone_controller = drone_controller

        self.subscription = self.create_subscription(
            PoseStamped,
            '/{}/pose'.format(drone_id),  # Adjusted to match the format in PART1, replace as necessary
            self.position_listener_callback,
            10)

    def position_listener_callback(self, msg):
        # Update drone_controller's current_position directly
    	self.drone_controller.position = np.append(self.drone_controller.position,[[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]],axis=0)
        
class DroneController:
    def __init__(self, drone_id, cf, starting_position, time_helper):
        self.drone_id = drone_id
        self.cf = cf  # Make sure this is correctly passed as an argument
        self.position = np.array([[0.3, 0, 0]])
        self.velocity = np.array([[0, 0, 0]])
        self.t_velocity = np.array([0])
        self.des_velocity = np.array([[0, 0, 0]])
        self.t_des_velocity = np.array([0])
        self.time_helper = time_helper  # Ensure this is passed and used correctly
        self.positions = []
        self.starting_position = starting_position

    def move_to_start_pos(self, duration):  # Remove unnecessary parameters
        self.cf.goTo(self.starting_position, 0.0, duration)

    def stop_setpoint(self):
        self.cf.notifySetpointsStop()

    def execute_trajectory(self, trajectory, duration):
        self.cf.uploadTrajectory(0, 0, trajectory)
        self.cf.startTrajectory(0, timescale=1.0)
        self.time_helper.sleep(duration)
        

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    
    z = 1.0
    
    drone1_id = 'cf_1'
    configs = [
        {'drone_id': drone1_id, 'starting_position': [0.0, 0.0, z]}
    ]
    Pos_subscribers = {}  # Dictionary to keep track of subscribers
    Vel_subscribers = {}
    drones = []
    x_des = np.array([[0, 0, 0.3]])
    
    for i, cf in enumerate(swarm.allcfs.crazyflies):
        drone_id = configs[i]['drone_id']
        drone = DroneController(drone_id, cf, configs[i]['starting_position'], timeHelper)
        Pos_subscribers[drone_id] = Position_Subscriber(drone_id, drone)
        drones.append(drone)

    for drone in drones:
        rclpy.spin_once(Pos_subscribers[drone.drone_id], timeout_sec=0.001)
        
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf.goTo([0.0,0.0,z], 0.0,2.0)
    timeHelper.sleep(3.0)
    cf.goTo([0.5,0.0,z+0.2], 0.0,2.0)
    for i in range(300):
    	time.sleep(0.025)
    	rclpy.spin_once(Pos_subscribers['cf_1'], timeout_sec=0.05)
    	x_des =  np.append(x_des,[[0.5,0.0,z+0.2]],axis=0)
    '''
    cf.goTo([0.0,0.0,0.4], 0.0,3.0)
    for i in range(200):
    	time.sleep(0.03)
    	rclpy.spin_once(Pos_subscribers['cf_1'], timeout_sec=0.05)
    	x_des =  np.append(x_des,[[0.0,0.0,0.4]],axis=0)
    '''
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)
    
    
    print(f'Steady-state error: {np.abs((drone.position[-1, 0] - 0.5) / 0.5 * 100):.2f}%')
    print(f'Overshoot: {np.abs((np.max(drone.position[:, 0]) - 0.5) / 0.5 * 100):.2f}%')
    for drone in drones:
    	plot_trajectory(drone.position, x_des)


if __name__ == '__main__':
    main()
