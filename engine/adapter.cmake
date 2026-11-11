if (NOT EXISTS "${JUQE_RUST_LIBRARY}")
    message(FATAL_ERROR "JUQE_RUST_LIBRARY must name the compiled Rust static library")
endif()

find_package(Threads REQUIRED)
add_library(juqe-rust STATIC IMPORTED)
set_target_properties(juqe-rust PROPERTIES IMPORTED_LOCATION "${JUQE_RUST_LIBRARY}")

set(JUQE_ADAPTER_SOURCES
    "${JUQE_ROOT}/adapters/llama/attach.cpp"
    "${JUQE_ROOT}/adapters/llama/callback.cpp"
    "${JUQE_ROOT}/adapters/llama/config.cpp"
    "${JUQE_ROOT}/adapters/llama/layout.cpp"
    "${JUQE_ROOT}/adapters/llama/pools.cpp"
    "${JUQE_ROOT}/adapters/llama/routing.cpp"
)
set_source_files_properties(${JUQE_ADAPTER_SOURCES} PROPERTIES
    SKIP_UNITY_BUILD_INCLUSION ON
    COMPILE_OPTIONS "-Werror"
)
target_sources(llama PRIVATE ${JUQE_ADAPTER_SOURCES})
target_include_directories(llama PRIVATE "${JUQE_ROOT}/include" "${JUQE_ROOT}/adapters/llama")
target_link_libraries(llama PRIVATE juqe-rust Threads::Threads ${CMAKE_DL_LIBS} m)
