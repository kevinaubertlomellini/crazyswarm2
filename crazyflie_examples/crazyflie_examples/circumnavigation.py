"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

# from pycrazyswarm import Crazyswarm

from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import Crazyswarm
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import math, time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0
t_aux_v1 = np.array([0])
t_aux_v2 = np.array([0])
t_aux_v3 = np.array([0])
t_aux_v4 = np.array([0])

t_aux_p1 = np.array([0])
t_aux_p2 = np.array([0])
t_aux_p3 = np.array([0])
t_aux_p4 = np.array([0])
t_aux = np.array([0])

drone1_id = 'cf_1'
drone2_id = 'cf_2'
drone3_id = 'cf_3'
drone4_id = 'cf_4' 

t_i = 0

factor = 0.6
k0 = 0.4*factor
k1 = 1.0*factor
k2 = 1.2*factor
k3 = 1.5*factor
n = 3
rho = 1.0

max_v = 0.25
iteraciones=1800
ptdot = np.array([[0.0], [0.0], [0.025]])
configs = [
        {'drone_id': drone1_id, 'starting_position': [0.1, 0.0, 0.75]},
        {'drone_id': drone2_id, 'starting_position': [-0.65, 0.73, 0.95]},
        {'drone_id': drone3_id, 'starting_position': [0.95, -0.75, 0.25]},
        {'drone_id': drone4_id, 'starting_position': [-0.3, -0.4, 0.5]}
    ]


drones_id = [drone1_id,drone2_id,drone3_id,drone4_id ]


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
        global t_aux_p1
        global t_aux_p2
        global t_aux_p3
        global t_aux_p4
        self.drone_controller.position = np.append(self.drone_controller.position,[[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]],axis=0)
        #print("---------------------------------------------")
        #print(msg)
        #print(f"{self.drone_id} position", round(msg.pose.position.x, 4),round(msg.pose.position.y, 4),round(msg.pose.position.z, 4))
        t_f=time.time()
        if self.drone_id == drone1_id:
            t_aux_p1= np.append(t_aux_p1,[t_f-t_i], axis=0)
        if self.drone_id == drone2_id:
            t_aux_p2= np.append(t_aux_p2,[t_f-t_i], axis=0) 
        if self.drone_id == drone3_id:
            t_aux_p3= np.append(t_aux_p3,[t_f-t_i], axis=0)
        if self.drone_id == drone4_id:
            t_aux_p4= np.append(t_aux_p4,[t_f-t_i], axis=0)


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
        global t_aux_v1
        global t_aux_v2
        global t_aux_v3
        global t_aux_v4
        self.drone_controller.velocity = np.append(self.drone_controller.velocity,[[msg.values[0], msg.values[1], msg.values[2]]], axis=0)
        t_f=time.time()
        if self.drone_id == drone1_id:
            t_aux_v1= np.append(t_aux_v1,[t_f-t_i], axis=0)
        if self.drone_id == drone2_id:
            t_aux_v2= np.append(t_aux_v2,[t_f-t_i], axis=0) 
        if self.drone_id == drone3_id:
            t_aux_v3= np.append(t_aux_v3,[t_f-t_i], axis=0)
        if self.drone_id == drone4_id:
            t_aux_v4= np.append(t_aux_v4,[t_f-t_i], axis=0)    
        #print("---------------------------------------------")
        #print(msg)
        #print(f"{self.drone_id} velocity", round(msg.values[0], 4), round(msg.values[1], 4), round(msg.values[2], 4))


def velocity_vector(pt, ptdot, i, p_i, p_1, p_2, p_3, dif, k):
    A = np.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    # Desired phasing angle between agents (thetaij may be different from thetaik)
    thetaij = 2 * math.pi / n
    # Desired inter-agent distance
    dij = 2 * rho * math.sin(thetaij / 2)
    # Matrix of interdv1-agent distances
    D = dij * A
    a = np.matrix([0, 0, 1]).transpose()
    a_a_transpose = a * a.transpose()
    pit = p_i - pt
    #print('pit',pit)
    DeliV1 = a_a_transpose * pit
    #print('DeliV1',DeliV1)
    # Orthogonal projection matrix of a
    Pa = np.eye(3, 3) - a_a_transpose
    phii = (pit / np.linalg.norm(pit))
    phiia = (Pa * phii) / np.linalg.norm(Pa * phii)
    DeliV2 = (np.linalg.norm(Pa * pit) - rho) * phiia
    DeliV3 = np.matrix([0.0, 0.0, 0.0])
    dif3 = 0
    for j in range(n):
        if j == 0:
            pj = p_1
        if j == 1:
            pj = p_2
            if i ==0:
                dif[0,k,2] = np.linalg.norm(p_i - pj)- D[i, j] # [pair(1-2),iteration,dif3]
        else:
            pj = p_3
            if i ==0:
                dif[1,k,2] = np.linalg.norm(p_i - pj)- D[i, j] # [pair(1-3),iteration,dif3]
            if i ==1:
                dif[2,k,2] = np.linalg.norm(p_i - pj)- D[i, j] # [pair(2-3),iteration,dif3]
        pjt = pj - pt
        phij = pjt / np.linalg.norm(pjt)
        phija = Pa * phij / np.linalg.norm(Pa * phij)
        gammaij = A[i, j] * ((np.linalg.norm(phiia - phija) ** 2) - ((D[i, j] ** 2) / (rho ** 2))) * np.cross(a.transpose(), phiia.transpose()) * (phiia - phija)
        DeliV3 += (1 / np.linalg.norm(Pa * pit)) * gammaij * np.cross(a.transpose(), phiia.transpose())
    ui1 = -k1 * DeliV1
    ui2 = (-k2 * DeliV2) + (k0 * np.linalg.norm(Pa * pit) * np.cross(a.transpose(), phiia.transpose()).transpose())
    ui3 = -k3 * np.linalg.norm(Pa * pit) * DeliV3.transpose()
    #print('ui1',ui1)
    #print('ui2',ui2)
    #print('ui3',ui3)
    #print('ptdot',ptdot)
    dif[i,k,0] = (a.transpose()*pit)  # [crazyflie(0-2),iteration,dif1]

    dif[i,k,1] = (np.linalg.norm(pit)-rho)  # [crazyflie(0-2),iteration,dif2]
    ui = ui1 + ui2 + ui3
    #print('ui',ui)
    return np.asarray(ui).flatten()


class DroneController:
    def __init__(self, drone_id, cf, starting_position, time_helper):
        self.drone_id = drone_id
        self.cf = cf  # Make sure this is correctly passed as an argument
        self.position = np.array([[0, 0, 0]])
        self.velocity = np.array([[0, 0, 0]])
        self.time_helper = time_helper  # Ensure this is passed and used correctly
        self.positions = []
        self.starting_position = starting_position

    def move_to_start_pos(self, duration):  # Remove unnecessary parameters
        self.cf.goTo(self.starting_position, 0.0, duration)

    def stop_setpoint(self):
        self.cf.notifySetpointsStop()


def control_circumnvigation(allcfs):
    global timeHelper
    global t_aux
    global t_i
    cf0 = allcfs.crazyflies[0]
    cf1 = allcfs.crazyflies[1]
    cf2 = allcfs.crazyflies[2]
    cf3 = allcfs.crazyflies[3]
    dv1_array = np.array([[0, 0, 0]])
    dv2_array = np.array([[0, 0, 0]])
    dv3_array = np.array([[0, 0, 0]])
    v1_array = np.array([[0, 0, 0]])
    v2_array = np.array([[0, 0, 0]])
    v3_array = np.array([[0, 0, 0]])

    

    dif  = np.zeros((3, iteraciones, 3))
    Pos_subscribers = {}  # Dictionary to keep track of subscribers
    Vel_subscribers = {}
    timeHelper.sleep(3.0)
    drones = []
    for i, cf in enumerate(allcfs.crazyflies):
        drone_id = configs[i]['drone_id']
        drone = DroneController(drone_id, cf, configs[i]['starting_position'], timeHelper)
        Pos_subscribers[drone_id] = Position_Subscriber(drone_id, drone)
        Vel_subscribers[drone_id] = Velocity_Subscriber(drone_id, drone)
        drones.append(drone)
    allcfs.takeoff(targetHeight=0.75, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    # Move all drones to their starting positions
    for drone in drones:
        drone.move_to_start_pos(5.0)

    timeHelper.sleep(6.0)

    t_i=time.time()

    for j, drone in enumerate(drones):
        rclpy.spin_once(Pos_subscribers[drone.drone_id], timeout_sec=0.1)
        rclpy.spin_once(Vel_subscribers[drone.drone_id], timeout_sec=0.1)
        if j == 0:
            p_cf0 = drone.position[-1, :]
        if j == 1:
            p_cf1 = drone.position[-1, :]
        if j == 2:
            p_cf2 = drone. position[-1, :]
        if j == 3:
            p_cf3 = drone.position[-1, :]
    t_i2=time.time()
    pt_z= 0.75
    pt_x= 0.0
    pt_y= 0.0
    for k in range(iteraciones):

        print(k)        
        pt = np.reshape(p_cf0, (3, 1))
        p_1 = np.reshape(p_cf1, (3, 1))
        p_2 = np.reshape(p_cf2, (3, 1))
        p_3 = np.reshape(p_cf3, (3, 1))
        v1 = velocity_vector(pt, ptdot, 0, p_1, p_1, p_2, p_3, dif, k)
        v2 = velocity_vector(pt, ptdot, 1, p_2, p_1, p_2, p_3, dif, k)
        v3 = velocity_vector(pt, ptdot, 2, p_3, p_1, p_2, p_3, dif, k)
        dv1_array = np.append(dv1_array, [[v1[0], v1[1], v1[2]]], axis=0)
        dv2_array = np.append(dv2_array, [[v2[0], v2[1], v2[2]]], axis=0)
        dv3_array = np.append(dv3_array, [[v3[0], v3[1], v3[2]]], axis=0)

        #print('v1',v1)
        #print('v2',v2)
        #print('v3',v3)
        
        if np.linalg.norm(v1) > max_v:
            v1 = v1 / (np.linalg.norm(v1) / max_v)
        if np.linalg.norm(v2) > max_v:
            v2 = v2 / (np.linalg.norm(v2) / max_v)
        if np.linalg.norm(v3) > max_v:
            v3 = v3 / (np.linalg.norm(v3) / max_v)
        '''
        v1 = np.clip(v1, -max_v, max_v)
        v2 = np.clip(v2, -max_v, max_v)
        v3 = np.clip(v3, -max_v, max_v)
        '''
        v1_array = np.append(v1_array, [[v1[0]+ptdot[0][0], v1[1]+ptdot[1][0], v1[2]+ptdot[2][0]]], axis=0)
        v2_array = np.append(v2_array, [[v2[0]+ptdot[0][0], v2[1]+ptdot[1][0], v2[2]+ptdot[2][0]]], axis=0)
        v3_array = np.append(v3_array, [[v3[0]+ptdot[0][0], v3[1]+ptdot[1][0], v3[2]+ptdot[2][0]]], axis=0)
        tf_2=time.time()
        pt_x=pt_x+ptdot[0][0]*(tf_2-t_i2)
        pt_y=pt_y+ptdot[1][0]*(tf_2-t_i2)
        pt_z=pt_z+ptdot[2][0]*(tf_2-t_i2)
        cf0.goTo([pt_x,pt_y,pt_z], 0.0,tf_2-t_i2)
        #print(tf_2-t_i2)
        #cf0.velWorld([0.0,0.0,0.75+0.005*(tf_2-t_i2)], 0.0)
        t_i2=tf_2
        cf1.velWorld([v1[0]+ptdot[0][0], v1[1]+ptdot[1][0], v1[2]+ptdot[2][0]], 0.0)
        cf2.velWorld([v2[0]+ptdot[0][0], v2[1]+ptdot[1][0], v2[2]+ptdot[2][0]], 0.0)
        cf3.velWorld([v3[0]+ptdot[0][0], v3[1]+ptdot[1][0], v3[2]+ptdot[2][0]], 0.0)
        t_f=time.time()
        t_aux= np.append(t_aux,[t_f-t_i], axis=0)
        for j, drone in enumerate(drones):
            # Process incoming messages for each drone to update their positions
            rclpy.spin_once(Pos_subscribers[drone.drone_id], timeout_sec=0.1)
            rclpy.spin_once(Vel_subscribers[drone.drone_id], timeout_sec=0.1)
            if j == 0:
                p_cf0 = drone.position[-1, :]
            if j == 1:
                p_cf1 = drone.position[-1, :]
            if j == 2:
                p_cf2 = drone.position[-1, :]
            if j == 3:
                p_cf3 = drone.position[-1, :]  
    for cf in allcfs.crazyflies:
        cf.notifySetpointsStop()
    allcfs.land(targetHeight=0.03, duration=8.0)
    time.sleep(10.0)
    for subscriber in Pos_subscribers.values():
        subscriber.destroy_node()

    for subscriber in Vel_subscribers.values():
        subscriber.destroy_node()
    rclpy.shutdown()
    plot_trajectories(drones)
    plot_velocities(drones,dv1_array,dv2_array,dv3_array,v1_array,v2_array,v3_array)
    plot_dif(dif)
    plt.show()

def plot_trajectories(drones):
    fig = plt.figure(figsize=(16,12))
    fig1 = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax1 = fig1.add_subplot(2,2,1)
    ax2 = fig1.add_subplot(2,2,2)
    ax3 = fig1.add_subplot(2,2,3)
    ax4 = fig1.add_subplot(2,2,4)

    subplots = [ax1, ax2, ax3, ax4]  # List of all subplots

    t_aux_p_vector = [t_aux_p1,t_aux_p2,t_aux_p3,t_aux_p4]

    for j, drone in enumerate(drones):
        positions = drone.position
        ax.plot(positions[1:, 0], positions[1:, 1], positions[1:, 2])
        subplots[j].plot(t_aux_p_vector[j][1:],positions[1:, 0], 'r', label='X')
        subplots[j].plot(t_aux_p_vector[j][1:],positions[1:, 1], 'g', label='Y')
        subplots[j].plot(t_aux_p_vector[j][1:],positions[1:, 2], 'b', label='Z')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_zlim(0.0,1.75)
    ax.set_xlim(-1.75,1.75)
    ax.set_ylim(-1.75,1.75)
    ax.legend(["Crazyflie 1","Crazyflie 2","Crazyflie 3","Crazyflie 4"])
    ax.set_title('Drone Trajectories')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Position of Crazyflie 1')
    ax1.grid()
    ax1.legend()

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position of Crazyflie 2')
    ax2.grid()
    ax2.legend()

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position of Crazyflie 3')
    ax3.grid()
    ax3.legend()

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position of Crazyflie 4')
    ax4.grid()
    ax4.legend()

    fig.savefig("3D_sim_fixed_1.png",dpi=750)
    fig1.savefig("Position_sim_fixed_1.png",dpi=750)

def plot_velocities(drones, dv1_array, dv2_array, dv3_array, v1_array, v2_array, v3_array):
    fig2A = plt.figure(figsize=(16,12))
    vx1 = fig2A.add_subplot(2, 2, 1)
    vx2 = fig2A.add_subplot(2, 2, 2)
    vx3 = fig2A.add_subplot(2, 2, 3)

    # Plot Crazyflie 1 velocities
    vx1.plot(t_aux,dv1_array[:, 0], 'r', label='computed_vX')
    vx1.plot(t_aux,dv1_array[:, 1], 'g', label='computed_vY')
    vx1.plot(t_aux,dv1_array[:, 2], 'b', label='computed_vZ')
    vx1.plot(t_aux,v1_array[:, 0], 'darkred', label='d_vX')
    vx1.plot(t_aux,v1_array[:, 1], 'darkslategrey', label='d_vY')
    vx1.plot(t_aux,v1_array[:, 2], 'k', label='d_vZ')

    # Plot Crazyflie 2 velocities
    vx2.plot(t_aux,dv2_array[:, 0], 'r', label='computed_vX')
    vx2.plot(t_aux,dv2_array[:, 1], 'g', label='computed_vY')
    vx2.plot(t_aux,dv2_array[:, 2], 'b', label='computed_vZ')
    vx2.plot(t_aux,v2_array[:, 0], 'darkred', label='d_vX')
    vx2.plot(t_aux,v2_array[:, 1], 'darkslategrey', label='d_vY')
    vx2.plot(t_aux,v2_array[:, 2], 'k', label='d_vZ')

    # Plot Crazyflie 3 velocities
    vx3.plot(t_aux,dv3_array[:, 0], 'r', label='computed_vX')
    vx3.plot(t_aux,dv3_array[:, 1], 'g', label='computed_vY')
    vx3.plot(t_aux,dv3_array[:, 2], 'b', label='computed_vZ')
    vx3.plot(t_aux,v3_array[:, 0], 'darkred', label='d_vX')
    vx3.plot(t_aux,v3_array[:, 1], 'darkslategrey', label='d_vY')
    vx3.plot(t_aux,v3_array[:, 2], 'k', label='d_vZ')

    vx1.set_title('Computed Velocity of Crazyflie 1')
    vx2.set_title('Computed Velocity of Crazyflie 2')
    vx3.set_title('Computed Velocity of Crazyflie 3')

    vx1.grid()
    vx2.grid()
    vx3.grid()

    vx1.set_xlabel('Time (s)')
    vx1.set_ylabel('Velocity (m/s)')
    vx2.set_xlabel('Time (s)')
    vx2.set_ylabel('Velocity (m/s)')
    vx3.set_xlabel('Time (s)')
    vx3.set_ylabel('Velocity (m/s)')

    vx1.legend()
    vx2.legend()
    vx3.legend()

    plt.tight_layout()

    fig2A.savefig("Velocity_computed_sim_fixed_1.png",dpi=750)

    fig2B = plt.figure(figsize=(16,12))
    vx0B = fig2B.add_subplot(2, 2, 1)
    vx1B = fig2B.add_subplot(2, 2, 2)
    vx2B = fig2B.add_subplot(2, 2, 3)
    vx3B = fig2B.add_subplot(2, 2, 4)

    subplots = [vx0B, vx1B, vx2B, vx3B]  # List of all subplots

    # Plot Crazyflie 1 velocities
    vx0B.axhline(y = ptdot[0][0], color = 'r', linestyle = 'dashed',label='d_vX') 
    vx0B.axhline(y = ptdot[1][0], color = 'g', linestyle = 'dashed',label='d_vY') 
    vx0B.axhline(y = ptdot[2][0], color = 'b', linestyle = 'dashed',label='d_vZ') 
    vx0B.set_title("Velocity of Crazyflie 1")

    # Plot Crazyflie 1 velocities
    vx1B.plot(t_aux,v1_array[:, 0], 'r', label='d_vX')
    vx1B.plot(t_aux,v1_array[:, 1], 'g', label='d_vY')
    vx1B.plot(t_aux,dv1_array[:, 2], 'b', label='d_vZ')
    vx1B.set_title("Velocity of Crazyflie 2")

    # Plot Crazyflie 2 velocities
    vx2B.plot(t_aux,v2_array[:, 0], 'r', label='d_vX')
    vx2B.plot(t_aux,v2_array[:, 1], 'g', label='d_vY')
    vx2B.plot(t_aux,v2_array[:, 2], 'b', label='d_vZ')
    vx2B.set_title("Velocity of Crazyflie 3")

    # Plot Crazyflie 3 velocities
    vx3B.plot(t_aux,v3_array[:, 0], 'r', label='d_vX')
    vx3B.plot(t_aux,v3_array[:, 1], 'g', label='d_vY')
    vx3B.plot(t_aux,v3_array[:, 2], 'b', label='d_vZ')
    vx2B.set_title("Velocity of Crazyflie 4")

    t_aux_v_vector = [t_aux_v1,t_aux_v2,t_aux_v3,t_aux_v4]

    for j, drone in enumerate(drones):
        velocities = drone.velocity
        subplots[j].plot(t_aux_v_vector[j][1:],velocities[1:, 0], 'darkred', label='vX')
        subplots[j].plot(t_aux_v_vector[j][1:],velocities[1:, 1], 'darkslategrey', label='vY')
        subplots[j].plot(t_aux_v_vector[j][1:],velocities[1:, 2], 'k', label='vZ')
        subplots[j].set_xlabel('Time (s)')
        subplots[j].set_ylabel('Velocity (m/s)')
        subplots[j].grid()
        subplots[j].legend()

    fig2B.savefig("Velocity_sim_fixed_1.png",dpi=750)

def plot_dif(dif):
    fig3 = plt.figure(figsize=(16,12))
    difx1 = fig3.add_subplot(2, 2, 1)
    difx2 = fig3.add_subplot(2, 2, 2)
    difx3 = fig3.add_subplot(2, 2, 3)

    # Plot Crazyflie 1 velocities
    difx1.plot(t_aux[1:],dif[0,:, 0], 'r', label='Crazyflie 1')
    difx1.plot(t_aux[1:],dif[1,:, 0], 'g', label='Crazyflie 2')
    difx1.plot(t_aux[1:],dif[2,:, 0], 'b', label='Crazyflie 3')

    # Plot Crazyflie 1 velocities
    difx2.plot(t_aux[1:],dif[0,:, 1], 'r', label='Crazyflie 1')
    difx2.plot(t_aux[1:],dif[1,:, 1], 'g', label='Crazyflie 2')
    difx2.plot(t_aux[1:],dif[2,:, 1], 'b', label='Crazyflie 3')

    # Plot Crazyflie 1 velocities
    difx3.plot(t_aux[1:],dif[0,:, 2], 'r', label='Pair (1,2)')
    difx3.plot(t_aux[1:],dif[1,:, 2], 'g', label='Pair (1,3)')
    difx3.plot(t_aux[1:],dif[2,:, 2], 'b', label='Pair (2,3)')

    difx1.set_title('Target plane convergence')
    difx2.set_title('Distance error between agent i and target')
    difx3.set_title('Inter-agent distance')

    difx1.grid()
    difx2.grid()
    difx3.grid()

    difx1.legend()
    difx2.legend()
    difx3.legend()

    difx1.set_xlabel('Time (s)')
    difx1.set_ylabel('Distance (m)')
    difx2.set_xlabel('Time (s)')
    difx2.set_ylabel('Distance (m)')
    difx3.set_xlabel('Time (s)')
    difx3.set_ylabel('Distance (m)')

    fig3.savefig("Error_sim_fixed_1.png",dpi=750)

def main(args=None) -> None:
    global timeHelper
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    try:
        control_circumnvigation(allcfs)
    except KeyboardInterrupt:
        print("Keyboard interrupt, triggering landing sequence")
        time.sleep(0.4)
        for cf in allcfs.crazyflies:
            cf.notifySetpointsStop()
        time.sleep(0.1)
        allcfs.land(targetHeight=0.03, duration=8.0)
        time.sleep(10.0)

if __name__ == "__main__":
    main()
