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
    cf0.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    cf1.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    for i in range(50):
        cf0.velWorld([0.25,0.0,0.0],0.0)
        cf1.velWorld([0.0,0.1,0.0],0.0)
        timeHelper.sleep(0.1)
    for i in range(60):
        cf0.velWorld([0.0,0.0,-0.1],0.0)
        cf1.velWorld([0.0,0.0,-0.1],0.0)
        timeHelper.sleep(0.1)

if __name__ == "__main__":
    main()
