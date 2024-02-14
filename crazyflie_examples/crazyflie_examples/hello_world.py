"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

# from pycrazyswarm import Crazyswarm
from crazyflie_py import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    print(123)
    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    print(123)
    #cf.goTo([1.0,1.0,1.0],0.0, duration=TAKEOFF_DURATION)
    for i in range(30):
        cf.velWorld([0.1,0.1,0.1],0.0)
        timeHelper.sleep(0.11)
        
if __name__ == "__main__":
    main()
