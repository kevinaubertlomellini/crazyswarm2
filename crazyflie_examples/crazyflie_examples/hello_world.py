"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm


TAKEOFF_DURATION = 5.0
HOVER_DURATION = 2.5


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs

    cf.takeoff(targetHeight=0.7, duration=TAKEOFF_DURATION)
    timeHelper.sleep(8.0)
    cf.goTo([0.0,0.0,0.2], 0.0,3.0)
    timeHelper.sleep(3.0)
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == '__main__':
    main()
