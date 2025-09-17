set(LIB srm_core)
message("Configuring coord module...")

aux_source_directory(src SRC)
add_library(${LIB} SHARED ${SRC})

target_include_directories(
        ${LIB}
        PUBLIC include
        PUBLIC ${PROJECT_SOURCE_DIR}/modules/video/include
        PUBLIC ${PROJECT_SOURCE_DIR}/modules/coord/include
        PUBLIC ${PROJECT_SOURCE_DIR}/modules/message/include
        PUBLIC ${PROJECT_SOURCE_DIR}/modules/viewer/include
)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  target_link_libraries(
          ${LIB}
          PUBLIC srm_common
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/video/lib/libsrm_video.dylib
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/coord/lib/libsrm_coord.dylib
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/message/lib/libsrm_message.dylib
          PUBLIC srm_autoaim
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/viewer/lib/libsrm_viewer.dylib
  )
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
  target_link_libraries(
          ${LIB}
          PUBLIC srm_common
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/video/lib/libsrm_video.so
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/coord/lib/libsrm_coord.so
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/message/lib/libsrm_message.so
          PUBLIC srm_autoaim
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/viewer/lib/libsrm_viewer.so
  )
endif ()
