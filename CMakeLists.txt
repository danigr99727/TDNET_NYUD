cmake_minimum_required(VERSION 2.8.3)
project(tdnet_nyud)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        )

catkin_package()

include_directories(
        ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
        segmentation.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )