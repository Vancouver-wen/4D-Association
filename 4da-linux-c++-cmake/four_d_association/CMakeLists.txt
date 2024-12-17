set(
    FOUR_D_ASSOCIATION_HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
file(
    GLOB 
    FOUR_D_ASSOCIATION_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp
)

# 生成 four_d_association 静态库
add_library(
    four_d_association 
    STATIC
    ${FOUR_D_ASSOCIATION_SOURCE_FILES}
)

target_include_directories(
    four_d_association
    PUBLIC 
    ${FOUR_D_ASSOCIATION_HEADER_FILES}
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)   

target_link_libraries(
    four_d_association
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)


# 定义 mocap 与 evaluate_shelf 两个可执行程序
add_executable(
    mocap
    main.cpp
)

add_executable(
    evaluate_shelf
    evaluate_shelf_main.cpp
)

target_link_libraries(
    mocap
    four_d_association
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
)

target_link_libraries(
    evaluate_shelf
    four_d_association
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
)
