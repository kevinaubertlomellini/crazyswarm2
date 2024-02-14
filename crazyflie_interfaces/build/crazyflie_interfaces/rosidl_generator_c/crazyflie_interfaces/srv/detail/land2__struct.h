// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from crazyflie_interfaces:srv/Land2.idl
// generated code does not contain a copyright notice

#ifndef CRAZYFLIE_INTERFACES__SRV__DETAIL__LAND2__STRUCT_H_
#define CRAZYFLIE_INTERFACES__SRV__DETAIL__LAND2__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'duration'
#include "builtin_interfaces/msg/detail/duration__struct.h"

/// Struct defined in srv/Land2 in the package crazyflie_interfaces.
typedef struct crazyflie_interfaces__srv__Land2_Request
{
  uint8_t group_mask;
  float height;
  builtin_interfaces__msg__Duration duration;
} crazyflie_interfaces__srv__Land2_Request;

// Struct for a sequence of crazyflie_interfaces__srv__Land2_Request.
typedef struct crazyflie_interfaces__srv__Land2_Request__Sequence
{
  crazyflie_interfaces__srv__Land2_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} crazyflie_interfaces__srv__Land2_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/Land2 in the package crazyflie_interfaces.
typedef struct crazyflie_interfaces__srv__Land2_Response
{
  uint8_t structure_needs_at_least_one_member;
} crazyflie_interfaces__srv__Land2_Response;

// Struct for a sequence of crazyflie_interfaces__srv__Land2_Response.
typedef struct crazyflie_interfaces__srv__Land2_Response__Sequence
{
  crazyflie_interfaces__srv__Land2_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} crazyflie_interfaces__srv__Land2_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CRAZYFLIE_INTERFACES__SRV__DETAIL__LAND2__STRUCT_H_
