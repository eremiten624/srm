set(LIB srm_autoaim)
message("Configuring coord module...")

aux_source_directory(src SRC)
aux_source_directory(src/armor SRC)
add_library(${LIB} SHARED ${SRC})

target_include_directories(
        ${LIB}
        PUBLIC include
        PUBLIC ${PROJECT_SOURCE_DIR}/modules/coord/include
        PUBLIC ${PROJECT_SOURCE_DIR}/modules/nn/include
        PUBLIC ${PROJECT_SOURCE_DIR}/modules/viewer/include
)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  target_link_libraries(
          ${LIB}
          PUBLIC srm_common
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/coord/lib/libsrm_coord.dylib
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/nn/lib/libsrm_nn.dylib
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/viewer/lib/libsrm_viewer.dylib
  )
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
  target_link_libraries(
          ${LIB}
          PUBLIC srm_common
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/coord/lib/libsrm_coord.so
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/nn/lib/libsrm_nn.so
          PUBLIC ${PROJECT_SOURCE_DIR}/modules/viewer/lib/libsrm_viewer.so
  )
endif ()
