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
    cf1 = swarm.allcfs.crazyflies[1]
    cf2 = swarm.allcfs.crazyflies[2]
    cf3 = swarm.allcfs.crazyflies[3]
    rclpy.spin_once(position_subscriber2)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf0.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf1.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf2.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf3.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    for i in range(60):
        cf0.velWorld([0.1,0.1,0.1],0.0)
        cf1.velWorld([0.2,0.2,0.2],0.0)
        cf2.velWorld([0.3,0.3,0.3],0.0)
        cf3.velWorld([0.4,0.4,0.4],0.0)
        timeHelper.sleep(0.1)
        rclpy.spin_once(position_subscriber2)
        print(cf0_position[2])
    
    for i in range(30):
        cf0.velWorld([0.0,0.0,0.0],0.0)
        cf1.velWorld([0.0,0.0,0.0],0.0)
        cf2.velWorld([0.0,0.0,0.0],0.0)
        cf3.velWorld([0.0,0.0,0.0],0.0)
        timeHelper.sleep(0.1)
        rclpy.spin_once(position_subscriber2)
        print(cf0_position[2])
    
    cf0.notifySetpointsStop()
    cf0.land(targetHeight=0.03, duration=5.0)
    timeHelper.sleep(0.01)
    cf1.notifySetpointsStop()
    cf1.land(targetHeight=0.03, duration=6.0)
    timeHelper.sleep(0.01)
    cf2.notifySetpointsStop()
    cf2.land(targetHeight=0.03, duration=7.0)
    timeHelper.sleep(0.01)
    cf3.notifySetpointsStop()
    cf3.land(targetHeight=0.03, duration=8.0)
    timeHelper.sleep(3.0)
        
    position_subscriber2.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
