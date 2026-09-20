if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    if(NOT BUILD_UCM_ASU)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]asu[/\\\\].*")
    endif()
    if(NOT TARGET dramstore)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]dram[/\\\\].*")
    endif()
    if(NOT TARGET yuanrongstore)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*/yuanrong/.*\\.cc$")
    endif()
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
        set_source_files_properties(case/dram/kv_protocol_test.cc
            PROPERTIES COMPILE_OPTIONS "-Wno-nonnull")
    endif()
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore fakestore pcstore healthbreakerstore
        gtest_main gtest gmock
    )
    if(TARGET dramstore AND TARGET drampool_core)
        target_link_libraries(ucmstore.test PRIVATE
            dramstore
            drampool_core
        )
    endif()
    if(TARGET yuanrongstore)
        target_compile_definitions(ucmstore.test PRIVATE UCM_YUANRONG_TESTING)
        target_link_libraries(ucmstore.test PRIVATE yuanrongstore)
    endif()
    if(BUILD_UCM_ASU)
        target_link_libraries(ucmstore.test PRIVATE
            kv_client
            buffer_pool
            trans
        )
        target_compile_definitions(ucmstore.test PRIVATE ASU_BUILD_TESTS)
    endif()
    target_link_libraries(ucmstore.test PRIVATE delegator delegator_store)
    gtest_discover_tests(ucmstore.test)
endif()

if(BUILD_UNIT_TESTS)
    set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/latest"
        CACHE PATH "Ascend toolkit root (symlink, version-agnostic)")
    set(ASCEND_DRV_LIB_PATH "/usr/local/Ascend/driver/lib64/driver"
        CACHE PATH "Ascend HAL driver library path")
    add_executable(cache2_copy_stream.test case/cache2/copy_stream_test.cc)
    target_include_directories(cache2_copy_stream.test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/case
        ${CMAKE_CURRENT_SOURCE_DIR}/../cache2/cc
        ${ASCEND_TOOLKIT_PATH}/aarch64-linux/pkg_inc/driver
        ${ASCEND_TOOLKIT_PATH}/aarch64-linux/include
    )
    target_link_directories(cache2_copy_stream.test PRIVATE
        ${ASCEND_DRV_LIB_PATH}
        ${ASCEND_TOOLKIT_PATH}/aarch64-linux/lib64
    )
    target_link_libraries(cache2_copy_stream.test PRIVATE
        cachestore2 trans gtest_main gtest ascend_hal ascendcl fmt pthread
    )
    gtest_discover_tests(cache2_copy_stream.test)
endif()
