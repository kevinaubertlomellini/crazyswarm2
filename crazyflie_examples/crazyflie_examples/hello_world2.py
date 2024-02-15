"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

# from pycrazyswarm import Crazyswarm
from crazyflie_py import Crazyswarm
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0
cf0_position=[0,0,0]

class PosSubscriber(Node):
    def __init__(self):
        super().__init__('position_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/cf_1/pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
    	global cf0_position
    	cf0_position[0]=msg.pose.position.x
    	cf0_position[1]=msg.pose.position.y
    	cf0_position[2]=msg.pose.position.z
    
def main():
    swarm = Crazyswarm()
    position_subscriber2 = PosSubscriber()
    rclpy.spin_once(position_subscriber2)
    timeHelper = swarm.timeHelper
    cf0 = swarm.allcfs.crazyflies[0]
    rclpy.spin_once(position_subscriber2)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf0.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    for i in range(50):
        cf0.velWorld([0.1,0.1,0.1],0.0)
        timeHelper.sleep(0.1)
        rclpy.spin_once(position_subscriber2)
        print(cf0_position)
    while cf0_position[2] > 0.04:
        cf0.velWorld([0.0,0.0,-0.2],0.0)
        timeHelper.sleep(0.1)
        rclpy.spin_once(position_subscriber2)
        print(cf0_position[2])
        
    position_subscriber2.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
