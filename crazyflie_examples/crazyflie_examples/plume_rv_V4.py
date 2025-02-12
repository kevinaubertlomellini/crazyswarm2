import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class PathPublisher(Node):

    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(Path, 'plume', 10)
        timer_period = 2.0  # segundos
        self.timer = self.create_timer(timer_period, self.publish_path)
        self.get_logger().info('Path publisher node started')

    def publish_path(self):
        path = Path()
        path.header.frame_id = 'world'  # Asegúrate de usar el frame correcto
        path.header.stamp = self.get_clock().now().to_msg()

        # Datos de ejemplo
        x = [0.7595, 0.7671, 0.7692, 0.7660, 0.7576, 0.7555, 0.7441, 0.7252, 0.7009, 0.6709, 0.6351, 0.6330,
             0.5924, 0.5428, 0.5106, 0.4853, 0.4185, 0.3881, 0.3409, 0.2657, 0.2506, 0.1434, 0.1432, 0.0208,
             0.0120, -0.1017, -0.1577, -0.2241, -0.3466, -0.4044, -0.4690, -0.5915, -0.7139, -0.8364, -0.9588,
             -1.0813, -1.2037, -1.3262, -1.3506, -1.4487, -1.5711, -1.6101, -1.6936, -1.7910, -1.8160, -1.9327,
             -1.9385, -2.0467, -2.0609, -2.1407, -2.1834, -2.2200, -2.2863, -2.3058, -2.3396, -2.3822, -2.4160,
             -2.4283, -2.4391, -2.4525, -2.4571, -2.4532, -2.4409, -2.4283, -2.4194, -2.3877, -2.3482, -2.3058,
             -2.3008, -2.2408, -2.1834, -2.1741, -2.0945, -2.0609, -2.0043, -1.9385, -1.9032, -1.8160, -1.7887,
             -1.6936, -1.6582, -1.5711, -1.5087, -1.4487, -1.3381, -1.3262, -1.2037, -1.1247, -1.0813, -0.9588,
             -0.8404, -0.8364, -0.7139, -0.5915, -0.4690, -0.3466, -0.2241, -0.1017, -0.0096, 0.0208, 0.1432,
             0.2013, 0.2657, 0.3359, 0.3881, 0.4356, 0.5106, 0.5136, 0.5766, 0.6272, 0.6330, 0.6686, 0.7016,
             0.7273, 0.7464, 0.7555, 0.7595]

        x = [i * 0.825 for i in x]

        y = [-0.2427, -0.3206, -0.3985, -0.4764, -0.5543, -0.5671, -0.6323, -0.7102, -0.7881, -0.8660,
             -0.9440, -0.9479, -1.0219, -1.0998, -1.1449, -1.1777, -1.2557, -1.2879, -1.3336, -1.3996,
             -1.4115, -1.4894, -1.4896, -1.5629, -1.5674, -1.6227, -1.6453, -1.6710, -1.7092, -1.7232,
             -1.7384, -1.7594, -1.7725, -1.7781, -1.7767, -1.7681, -1.7525, -1.7294, -1.7232, -1.6991,
             -1.6605, -1.6453, -1.6133, -1.5674, -1.5557, -1.4894, -1.4862, -1.4115, -1.4014, -1.3336,
             -1.2951, -1.2557, -1.1777, -1.1524, -1.0998, -1.0219, -0.9440, -0.9063, -0.8660, -0.7881,
             -0.7102, -0.6323, -0.5543, -0.5082, -0.4764, -0.3985, -0.3206, -0.2508, -0.2427, -0.1647,
             -0.0975, -0.0868, -0.0089, 0.0203, 0.0690, 0.1200, 0.1470, 0.2066, 0.2249, 0.2821, 0.3028,
             0.3488, 0.3807, 0.4087, 0.4587, 0.4636, 0.5089, 0.5366, 0.5506, 0.5847, 0.6145, 0.6154,
             0.6362, 0.6515, 0.6601, 0.6606, 0.6524, 0.6346, 0.6145, 0.6057, 0.5614, 0.5366, 0.5009,
             0.4587, 0.4185, 0.3807, 0.3059, 0.3028, 0.2249, 0.1470, 0.1358, 0.0690, -0.0089, -0.0868,
             -0.1647, -0.2194, -0.2427]
        y = [i * 0.997 for i in y]

        # Define la altura constante
        altura = 1.0
        
        x_offset = [xi + 1.0 for xi in x]
        
        y_offset = [yi - 1 for yi in y]


        # Define los puntos del path
        poses = []
        for xi, yi in zip(x_offset, y_offset):
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = xi
            pose.pose.position.y = yi
            pose.pose.position.z = altura
            poses.append(pose)

        path.poses = poses
        self.publisher_.publish(path)
        self.get_logger().info('Publishing path with {} poses'.format(len(path.poses)))


def main(args=None):
    rclpy.init(args=args)
    path_publisher = PathPublisher()
    rclpy.spin(path_publisher)
    path_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
