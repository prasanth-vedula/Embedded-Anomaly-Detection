cmake_minimum_required(VERSION 3.20)
set(CMAKE_OBJECT_PATH_MAX 128)

project(EmbeddedAnomalyDetection
    VERSION 1.0.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(EMBEDDED_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

set(EMBEDDED_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc
)

set(EMBEDDED_CORE_SOURCES
    Src/HAL/hal.c
    Src/Sensor/sensor.c
    Src/Preprocess/preprocess.c
    Src/Window/window.c
    Src/TinyML/model_weights.c
    Src/TinyML/tinyml.c
    Src/Anomaly/anomaly.c
    Src/Event/event.c
    Src/Telemetry/telemetry.c
    Src/Config/deployment_parameters.c
)

add_executable(
    embedded_anomaly_detection
    Src/main.c
    ${EMBEDDED_CORE_SOURCES}
)

target_include_directories(
    embedded_anomaly_detection
    PRIVATE
        ${EMBEDDED_INCLUDE_DIR}
)

target_compile_options(
    embedded_anomaly_detection
    PRIVATE
        ${EMBEDDED_WARNING_FLAGS}
)

if(NOT MSVC)
    target_link_libraries(
        embedded_anomaly_detection
        PRIVATE
            m
    )
endif()


# ---------------------------------------------------------------------------
# Embedded unit/integration tests
# ---------------------------------------------------------------------------

enable_testing()

add_executable(
    embedded_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/embedded/test_embedded.c
    ${EMBEDDED_CORE_SOURCES}
)

target_include_directories(
    embedded_test
    PRIVATE
        ${EMBEDDED_INCLUDE_DIR}
)

target_compile_options(
    embedded_test
    PRIVATE
        ${EMBEDDED_WARNING_FLAGS}
)

if(NOT MSVC)
    target_link_libraries(
        embedded_test
        PRIVATE
            m
    )
endif()

add_test(
    NAME EmbeddedCoreTest
    COMMAND embedded_test
)


add_executable(
    embedded_preprocess_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/embedded/test_preprocess.c
    Src/Sensor/sensor.c
    Src/Preprocess/preprocess.c
    Src/Config/deployment_parameters.c
)

target_include_directories(
    embedded_preprocess_test
    PRIVATE
        ${EMBEDDED_INCLUDE_DIR}
)

target_compile_options(
    embedded_preprocess_test
    PRIVATE
        ${EMBEDDED_WARNING_FLAGS}
)

add_test(
    NAME EmbeddedPreprocessTest
    COMMAND embedded_preprocess_test
)


add_executable(
    embedded_tinyml_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/embedded/test_tinyml.c
    Src/TinyML/model_weights.c
    Src/TinyML/tinyml.c
)

target_include_directories(
    embedded_tinyml_test
    PRIVATE
        ${EMBEDDED_INCLUDE_DIR}
)

target_compile_options(
    embedded_tinyml_test
    PRIVATE
        ${EMBEDDED_WARNING_FLAGS}
)

add_test(
    NAME EmbeddedTinyMLTest
    COMMAND embedded_tinyml_test
)
