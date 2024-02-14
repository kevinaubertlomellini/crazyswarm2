// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from crazyflie_interfaces:msg/AttitudeSetpoint.idl
// generated code does not contain a copyright notice

#ifndef CRAZYFLIE_INTERFACES__MSG__DETAIL__ATTITUDE_SETPOINT__STRUCT_HPP_
#define CRAZYFLIE_INTERFACES__MSG__DETAIL__ATTITUDE_SETPOINT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__crazyflie_interfaces__msg__AttitudeSetpoint __attribute__((deprecated))
#else
# define DEPRECATED__crazyflie_interfaces__msg__AttitudeSetpoint __declspec(deprecated)
#endif

namespace crazyflie_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct AttitudeSetpoint_
{
  using Type = AttitudeSetpoint_<ContainerAllocator>;

  explicit AttitudeSetpoint_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->roll = 0.0f;
      this->pitch = 0.0f;
      this->yaw_rate = 0.0f;
      this->thrust = 0;
    }
  }

  explicit AttitudeSetpoint_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->roll = 0.0f;
      this->pitch = 0.0f;
      this->yaw_rate = 0.0f;
      this->thrust = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _roll_type =
    float;
  _roll_type roll;
  using _pitch_type =
    float;
  _pitch_type pitch;
  using _yaw_rate_type =
    float;
  _yaw_rate_type yaw_rate;
  using _thrust_type =
    uint16_t;
  _thrust_type thrust;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__roll(
    const float & _arg)
  {
    this->roll = _arg;
    return *this;
  }
  Type & set__pitch(
    const float & _arg)
  {
    this->pitch = _arg;
    return *this;
  }
  Type & set__yaw_rate(
    const float & _arg)
  {
    this->yaw_rate = _arg;
    return *this;
  }
  Type & set__thrust(
    const uint16_t & _arg)
  {
    this->thrust = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator> *;
  using ConstRawPtr =
    const crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__crazyflie_interfaces__msg__AttitudeSetpoint
    std::shared_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__crazyflie_interfaces__msg__AttitudeSetpoint
    std::shared_ptr<crazyflie_interfaces::msg::AttitudeSetpoint_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AttitudeSetpoint_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->roll != other.roll) {
      return false;
    }
    if (this->pitch != other.pitch) {
      return false;
    }
    if (this->yaw_rate != other.yaw_rate) {
      return false;
    }
    if (this->thrust != other.thrust) {
      return false;
    }
    return true;
  }
  bool operator!=(const AttitudeSetpoint_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AttitudeSetpoint_

// alias to use template instance with default allocator
using AttitudeSetpoint =
  crazyflie_interfaces::msg::AttitudeSetpoint_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace crazyflie_interfaces

#endif  // CRAZYFLIE_INTERFACES__MSG__DETAIL__ATTITUDE_SETPOINT__STRUCT_HPP_
