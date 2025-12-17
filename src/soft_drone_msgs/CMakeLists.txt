cmake_minimum_required(VERSION 3.8)
project(soft_drone_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# 生成自定义消息（核心：替换了你错误写入的sed命令）
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ReadDJIRC.msg"
  "msg/WriteDSHOT.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

# 测试相关（保留原有逻辑，无修改）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

