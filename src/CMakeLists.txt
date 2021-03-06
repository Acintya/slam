#add pcl
FIND_PACKAGE (PCL REQUIRED COMPONENTS common io)

#add opencv
FIND_PACKAGE (OpenCV 2 REQUIRED)

#sophus
FIND_PACKAGE (Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

#add header and lib
ADD_DEFINITIONS (${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES (${PCL_INCLUDE_DIRS})
LINK_LIBRARIES (${PCL_LIBRARY_DIRS})

list (REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE (generate_pointcloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES (generate_pointcloud 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

ADD_EXECUTABLE (detectFeatures detectFeatures.cpp)
TARGET_LINK_LIBRARIES (detectFeatures 
    slambase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

ADD_EXECUTABLE (useSophus useSophus.cpp)
TARGET_LINK_LIBRARIES (useSophus ${Sophus_LIBRARIES})