import rx
import math
from crazyflie_examples.sketch import DroneStreamFactory, SketchAction, LatLon, Plume, CO2, \
    plot_plumes, calculate_co2, calculate_turn_center, rotate_vector,\
    EARTH_CIRCUMFERENCE, FORWARD, TURN, ReadingPosition, unitary, latlon_plus_meters
from rx.subject import Subject, BehaviorSubject
import rx.operators as ops
import matplotlib.pyplot as plt
from crazyflie_examples.plot_util import *
import time
from datetime import datetime, timedelta
import pandas as pd

from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import Crazyswarm
import rclpy
import os
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import math, time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, Point
from geopy.distance import geodesic
from matplotlib.patches import Polygon as MPolygon

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 2.0

delta_time =0.6

#EARTH_CIRCUMFERENCE = 40008000
EARTH_CIRCUMFERENCE = 40075000

t_inicial = 0

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
        #print("---------------------------------------------")
        #print(msg)
        print(f"{self.drone_id} position", round(msg.pose.position.x, 4),round(msg.pose.position.y, 4),round(msg.pose.position.z, 4))


class Velocity_Subscriber(Node):
    def __init__(self, drone_id, drone_controller):
        super().__init__('vel_subscriber_{}'.format(drone_id))
        self.drone_id = drone_id
        self.drone_controller = drone_controller

        # Subscription to custom Velocity topic
        self.velocity_subscription = self.create_subscription(
            LogDataGeneric,
            '/{}/velocity'.format(drone_id),
            self.velocity_listener_callback,
            10)

    def velocity_listener_callback(self, msg):
        # Update drone_controller's current_velocity based on the custom velocity topic
        self.drone_controller.velocity = np.append(self.drone_controller.velocity,[[msg.values[0], msg.values[1], msg.values[2]]], axis=0)
        #print("---------------------------------------------")
        #print(msg)
        #print(f"{self.drone_id} velocity", round(msg.values[0], 4), round(msg.values[1], 4), round(msg.values[2], 4))
        tf = time.time()
        self.drone_controller.t_velocity = np.append(self.drone_controller.t_velocity,[tf-t_inicial], axis=0)


def add_ruler(plt, ax, length):
    height_scale = 0.7
    lowerleft = [plt.xlim()[0], plt.ylim()[0]]
    upperright = [plt.xlim()[1], plt.ylim()[1]]

    # Calculate width by latitide
    width = abs(1.0 * length / (EARTH_CIRCUMFERENCE / 360))
    height = (upperright[1] - lowerleft[1]) * 0.018 * height_scale

    location = [plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * .05,
                plt.ylim()[1] - ((plt.ylim()[1] - plt.ylim()[0]) * (0.05 + (height_scale * .04)))]

    ax.add_patch(Rectangle(location, width, height, ec=(0, 0, 0, 1), fc=(1, 1, 1, 1), lw=height_scale))
    ax.add_patch(Rectangle(location, width / 2, height, ec=(0, 0, 0, 1), fc=(0, 0, 0, 1), lw=height_scale))
    ax.annotate("0", xy=(location[0], location[1] + (1.5 * height)), ha='center')
    ax.annotate("{} m".format(length), xy=(location[0] + width, location[1] + (1.5 * height)), ha='center')


class Drone:

    def __init__(self, position, max_acceleration_value=0.05):
        self.start_lat_lon = position
        self.position = (0, 0, 0)
        self.velocity = (0, 0, 0)
        self.acceleration = (0, 0, 0)
        self.max_acceleration = [max_acceleration_value, max_acceleration_value]
        self.max_velocity = 0.15
        self.dt = delta_time

    def update_velocity(self, desired_velocity, cf,uav,aux):
        desired_acceleration = tuple((dv - v) / self.dt for v, dv in zip(self.velocity, desired_velocity))
        
        '''
        acceleration_magnitude = np.linalg.norm(desired_acceleration)
        if acceleration_magnitude > self.max_acceleration[uav]:
            desired_acceleration = tuple(
                a / acceleration_magnitude * self.max_acceleration[uav] for a in desired_acceleration)
        print('Desired acc', desired_acceleration)
        self.velocity = tuple(v + a * self.dt for v, a in zip(self.velocity, desired_acceleration))
        cf.velWorld([self.velocity[0], self.velocity[1], 0.0], 0.0)
        '''
        
        self.velocity=desired_velocity
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > self.max_velocity:
            self.velocity = tuple(
                v / velocity_magnitude * self.max_velocity for v in self.velocity)
        
        #cf.velWorld([self.velocity[0], self.velocity[1], 0.0], 0.0)
        desired_position = tuple(float(v) + float(a) * self.dt for v, a in zip(aux, self.velocity))
        print([desired_position[0],desired_position[1],1.0])
        '''
        
        '''
        if uav==0:
            altitud=0.92
        else:
            altitud=1.1
        cf.goTo([desired_position[0],desired_position[1],altitud], 0.0, delta_time)

        '''
        print('Desired vel',self.velocity)
        print('Position', cf_position)
        '''
        return self.velocity

    def update_position(self, cf_position, crazyflie):

        if crazyflie == 1:
            longitude = self.start_lat_lon.longitude + ((cf_position[0]-0.8) / (
                    math.cos(self.start_lat_lon.latitude * 0.01745) * (EARTH_CIRCUMFERENCE / 360)))
            latitude = self.start_lat_lon.latitude + ((cf_position[1] + 1.0) / (EARTH_CIRCUMFERENCE / 360))
        else:
            longitude = self.start_lat_lon.longitude + ((cf_position[0]-1.2) / (
                    math.cos(self.start_lat_lon.latitude * 0.01745) * (EARTH_CIRCUMFERENCE / 360)))
            latitude = self.start_lat_lon.latitude + ((cf_position[1] + 0.85) / (EARTH_CIRCUMFERENCE / 360))

        return LatLon(latitude, longitude)

class DroneController:
    def __init__(self, drone_id, cf, starting_position, time_helper):
        self.drone_id = drone_id
        self.cf = cf  # Make sure this is correctly passed as an argument
        self.position = np.array([[0, 0, 0]])
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


def run_sketch(df1p, df2p, offset, lambda_value, plumes, threshold, iterations, base_file_name=None,
               starting_direction=None, latest_gradient=None, zoom=None):
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    global t_inicial
    df1v = (0, 0, 0)
    df2v = (0, 0, 0)
    streamFactory = DroneStreamFactory()
    announceStream = Subject()
    df1SetVelocity = BehaviorSubject([0, 1])
    df2SetVelocity = BehaviorSubject([0, 1])
    sketchSubject = Subject()
    vector1Publisher = Subject()
    vector2Publisher = Subject()
    df1Position = Subject()
    df1Velocity = Subject()
    df2Position = Subject()
    df2Velocity = Subject()
    df1co2 = Subject()
    df2co2 = Subject()
    streamFactory.put_drone("DF1", df1Position, df1Velocity, df1co2)
    streamFactory.put_drone("DF2", df2Position, df1Velocity, df2co2)

    action1 = SketchAction("DF1", df1SetVelocity, announceStream, offset, lambda_value, "DF2", True, threshold,
                           streamFactory, sketchSubject, vector1Publisher, starting_direction, latest_gradient)
    action2 = SketchAction("DF2", df2SetVelocity, announceStream, offset, lambda_value, "DF1", False, threshold,
                           streamFactory, sketchSubject, vector2Publisher, starting_direction, latest_gradient)

    action1.step()
    action2.step()

    df1 = Drone(df1p)
    df2 = Drone(df2p)

    df1latitudes = []
    df2latitudes = []
    df1longitudes = []
    df2longitudes = []

    unique_tokens = set()

    scale = 0.012
    arrow_head_width = 0.00002*scale
    arrow_head_height = arrow_head_width
    arrow_width = 0.000002*scale
    arrow_length = 0.00005*scale

    cf0 = allcfs.crazyflies[0]
    cf1 = allcfs.crazyflies[1]
    drone1_id = 'cf_2'
    drone2_id = 'cf_4'

    timeHelper.sleep(1.0)
    configs = [
        {'drone_id': drone1_id, 'starting_position': [0.8, -1.0, 1.0]},
        {'drone_id': drone2_id, 'starting_position': [1.2, -0.85, 1.0]}
    ]
    Pos_subscribers = {}  # Dictionary to keep track of subscribers
    Vel_subscribers = {}
    drones = []
    for i, cf in enumerate(allcfs.crazyflies):
        drone_id = configs[i]['drone_id']
        drone = DroneController(drone_id, cf, configs[i]['starting_position'], timeHelper)
        Pos_subscribers[drone_id] = Position_Subscriber(drone_id, drone)
        Vel_subscribers[drone_id] = Velocity_Subscriber(drone_id, drone)
        drones.append(drone)
    

    t_inicial = time.time()

    for drone in drones:
        rclpy.spin_once(Pos_subscribers[drone.drone_id], timeout_sec=0.1)
        rclpy.spin_once(Vel_subscribers[drone.drone_id], timeout_sec=0.1)

    def add_algorithm_details(token):
        #         if hasattr(token, 'vector_to_crossing') and hasattr(token, 'updated_position'):
        #             plt.arrow(token.updated_position.longitude, token.updated_position.latitude, token.vector_to_crossing[0] * arrow_length, token.vector_to_crossing[1] * arrow_length, head_width=arrow_head_width, head_length=arrow_head_height, width=arrow_width, fc='m', ec='m')
        #             [dx, dy] = rotate_vector([token.x, token.y], (token.a * token.p))

        #             plt.arrow(token.updated_position.longitude, token.updated_position.latitude, dx * arrow_length, dy * arrow_length, head_width=arrow_head_width, head_length=arrow_head_height, width=arrow_width, fc='b', ec='b')
        if not token in unique_tokens:
            if token.movement == FORWARD:
                plt.arrow(token.position.longitude, token.position.latitude, token.x * arrow_length,
                          token.y * arrow_length, head_width=arrow_head_width, head_length=arrow_head_height,
                          width=arrow_width, fc='k', ec='k')
            if token.movement == TURN:
                center = calculate_turn_center(token, offset * lambda_value / math.sqrt(lambda_value))
                plt.scatter(center.longitude, center.latitude, color='k', marker='+')
                plt.arrow(token.position.longitude, token.position.latitude, token.x * arrow_length,
                          token.y * arrow_length, head_width=arrow_head_width, head_length=arrow_head_height,
                          width=arrow_width, fc='b', ec='b')

            if hasattr(token, 'gradient'):
                plt.arrow(token.position.longitude, token.position.latitude, token.gradient[0] * 0.00005*scale,
                          token.gradient[1] * 0.00005*scale, head_width=0.00002*scale, head_length=0.00002*scale, width=0.000002*scale, fc='r',
                          ec='r')

            #             if hasattr(token, 'prev_pt'):
            #                 print("prev5_pt")
            #                 plt.scatter(token.prev_pt.longitude, token.prev_pt.latitude, color='k', s=5)

            unique_tokens.add(token)

    sketchSubject.subscribe(on_next=add_algorithm_details)

    plt.figure(1,figsize=(10, 8))
    plt.ticklabel_format(style='plain', useOffset=False)

    experiment_directory= plot_plumes(plt, threshold, lon=[-106.59725, -106.59714], lat=[35.19725, 35.19732], plumes=plumes)

    for plume in plumes:
        plt.plot(plume.source.longitude, plume.source.latitude, marker='*', c='r', markeredgewidth=1,
                 markeredgecolor=(0, 0, 0, 1), markersize=10)

    starting_time = datetime(2020,1,1)
    
    for drone in drones:
            rclpy.spin_once(Vel_subscribers[drone.drone_id], timeout_sec=0.1)
            rclpy.spin_once(Pos_subscribers[drone.drone_id], timeout_sec=0.1)

    try:
        timeHelper.sleep(2.0)
        allcfs.takeoff(targetHeight=0.5, duration=TAKEOFF_DURATION)
        timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)

        # Move all drones to their starting positions
        for drone in drones:
            drone.move_to_start_pos(4.0)

        timeHelper.sleep(7.0)
        for i in range(iterations):
            # print(i)
            for drone in drones:
                rclpy.spin_once(Vel_subscribers[drone.drone_id], timeout_sec=0.1)
                rclpy.spin_once(Pos_subscribers[drone.drone_id], timeout_sec=0.1)
                if drone.drone_id == drone1_id:
                    df1v = [0,0]
                    aux1 = drone.position[-1, :]
                    df1p = df1.update_position(drone.position[-1, :], 1)
                elif drone.drone_id == drone2_id:
                    df2v = [0,0]
                    df2p = df2.update_position(drone.position[-1, :], 2)
                    aux2 = drone.position[-1, :]

            current_time = (starting_time + timedelta(seconds=i * 0.1))
            df1co2v = calculate_co2(df1p, plumes)
            df2co2v = calculate_co2(df2p, plumes)
            df1Position.on_next([df1p, current_time])
            df1Velocity.on_next([df1v, current_time])
            df1co2.on_next([CO2(df1co2v), current_time])
            df2Position.on_next([df2p, current_time])
            df2Velocity.on_next([df2v, current_time])
            df2co2.on_next([CO2(df2co2v), current_time])

            print(i)

            for drone in drones:
                if drone.drone_id == drone1_id:
                    df1v = df1.update_velocity(np.array(df1SetVelocity.pipe(ops.first()).run()) /45, cf0,0,aux1)
                    drone.des_velocity = np.append(drone.des_velocity,[[df1v[0], df1v[1], 0]], axis=0)
                    tf = time.time()
                    drone.t_des_velocity = np.append(drone.t_des_velocity, [tf - t_inicial], axis=0)
                elif drone.drone_id == drone2_id:
                    df2v = df2.update_velocity(np.array(df2SetVelocity.pipe(ops.first()).run()) /45, cf1,1,aux2)
                    drone.des_velocity = np.append(drone.des_velocity, [[df2v[0], df2v[1], 0]], axis=0)
                    tf = time.time()
                    drone.t_des_velocity = np.append(drone.t_des_velocity, [tf - t_inicial], axis=0)

            df1latitudes.append(df1p.latitude)
            df1longitudes.append(df1p.longitude)
            df2latitudes.append(df2p.latitude)
            df2longitudes.append(df2p.longitude)
            timeHelper.sleep(delta_time+0.1)

    except KeyboardInterrupt:
        print("Keyboard interrupt, triggering landing sequence")

    time.sleep(0.1)
    cf0.notifySetpointsStop()
    cf1.notifySetpointsStop()
    time.sleep(0.05)
    allcfs.land(targetHeight=0.03, duration=5.0)
    time.sleep(6.0)

    plt.scatter(df1longitudes, df1latitudes, s=4)
    plt.scatter(df2longitudes, df2latitudes, s=4)
    plt.axis('equal')
    plt.grid()

    add_ruler(plt, plt.gca(), 0.5)
    
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

    if zoom is not None:
        plt.xlim(zoom[1])  # -106.5960, -106.5958
        plt.ylim(zoom[0])  # 35.1959, 35.1963


    base_directory = experiment_directory

    if base_file_name is not None:
        plt.savefig(f'{experiment_directory}/python_sim_small3.pdf', format="pdf", dpi=1000)

    # Ensure the experiment directory exists
    os.makedirs(experiment_directory, exist_ok=True)

    filename = os.path.join(experiment_directory, f"drone1_latitude_trajectory.csv")

    df = pd.DataFrame(df1latitudes)
    df.to_csv(filename, index=False)

    filename = os.path.join(experiment_directory, f"drone2_latitude_trajectory.csv")

    df = pd.DataFrame(df2latitudes)
    df.to_csv(filename, index=False)

    filename = os.path.join(experiment_directory, f"drone1_longitude_trajectory.csv")

    df = pd.DataFrame(df1longitudes)
    df.to_csv(filename, index=False)

    filename = os.path.join(experiment_directory, f"drone2_longitude_trajectory.csv")

    df = pd.DataFrame(df2longitudes)
    df.to_csv(filename, index=False) 

    for subscriber in Pos_subscribers.values():
        subscriber.destroy_node()

    for subscriber in Vel_subscribers.values():
        subscriber.destroy_node()
    plt.show()

    rclpy.shutdown()

def main(args=None) -> None:
    plumes = [
        Plume(LatLon(35.1973, -106.5972), -40, 12000, 2.0)
    ]

    threshold = 680

    df1p = LatLon(35.1972855, -106.5971789)
    df2p = LatLon(35.1972855, -106.5971764)

    run_sketch(df1p, df2p, 0.22, 0.001, plumes, threshold, iterations=1000, base_file_name='single_plume')

if __name__ == "__main__":
    main()
