"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

# from pycrazyswarm import Crazyswarm
from crazyflie_py import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf0 = swarm.allcfs.crazyflies[0]
    cf1 = swarm.allcfs.crazyflies[1]
    cf2 = swarm.allcfs.crazyflies[2]
    cf3 = swarm.allcfs.crazyflies[3]
    cf4 = swarm.allcfs.crazyflies[4]
    cf5 = swarm.allcfs.crazyflies[5]
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf0.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf1.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf2.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf3.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf4.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf5.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    for i in range(50):
        cf0.velWorld([0.0,0.1,0.1],0.0)
        cf1.velWorld([0.0,0.1,0.2],0.0)
        cf2.velWorld([0.0,0.1,0.3],0.0)
        cf3.velWorld([0.0,0.1,0.4],0.0)
        cf4.velWorld([0.0,0.1,0.5],0.0)
        cf5.velWorld([0.0,0.1,0.6],0.0)
        timeHelper.sleep(0.1)
    for i in range(50):
        cf0.velWorld([0.0,0.0,-0.25],0.0)
        cf1.velWorld([0.0,0.0,-0.35],0.0)
        cf2.velWorld([0.0,0.0,-0.45],0.0)
        cf3.velWorld([0.0,0.0,-0.55],0.0)
        cf4.velWorld([0.0,0.0,-0.65],0.0)
        cf5.velWorld([0.0,0.0,-0.75],0.0)
        timeHelper.sleep(0.1)

if __name__ == "__main__":
    main()
