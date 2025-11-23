# Makefile for Monte Carlo Risk Simulation

# Compiler and flags
CC = gcc
CFLAGS = -Wall -O3 -march=native -g
# macOS clang support: use -Xclang -fopenmp for clang, -fopenmp for GCC
OMP_FLAGS = -fopenmp
LIBOMP_PATH = $(shell brew --prefix libomp 2>/dev/null)
OMP_INCLUDE = $(if $(LIBOMP_PATH),-I$(LIBOMP_PATH)/include,)
OMP_LIBPATH = $(if $(LIBOMP_PATH),-L$(LIBOMP_PATH)/lib,)
OMP_LIB = $(if $(LIBOMP_PATH),-lomp,)
GSL_PATH = $(shell pkg-config --cflags-only-I gsl 2>/dev/null || echo "-I/usr/local/include -I/opt/homebrew/include")
GSL_LIBPATH = $(shell pkg-config --libs-only-L gsl 2>/dev/null || echo "-L/usr/local/lib -L/opt/homebrew/lib")
GSL_FLAGS = -lgsl -lgslcblas -lm
# YAML_FLAGS will be set by environment when module is loaded
# But we provide a fallback in case it's not set
YAML_FLAGS = -lyaml
YAML_INCLUDE = $(shell if [ -n "$$EBROOTYAML" ]; then echo "-I$$EBROOTYAML/include"; else pkg-config --cflags yaml-0.1 2>/dev/null || echo "-I/usr/local/include -I/opt/homebrew/include"; fi)
YAML_LIBPATH = $(shell if [ -n "$$EBROOTYAML" ]; then echo "-L$$EBROOTYAML/lib"; else pkg-config --libs-only-L yaml-0.1 2>/dev/null || echo "-L/usr/local/lib -L/opt/homebrew/lib"; fi)

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SERIAL_SRC = $(SRC_DIR)/02-C-serial/monte_carlo_serial.c
OMP_SRC = $(SRC_DIR)/02-openMP/monte_carlo_omp.c
OMP_OPT_SRC = $(SRC_DIR)/02-OptimizedOpenMP/monte_carlo_opt_omp.c
MAIN_SRC = $(SRC_DIR)/main_runner.c
UTIL_SRC = $(SRC_DIR)/utilities/load_binary.c
CONFIG_SRC = $(SRC_DIR)/utilities/load_config.c
CSV_SRC = $(SRC_DIR)/utilities/csv_writer.c

# Object files
OBJS = $(BUILD_DIR)/main_runner.o $(BUILD_DIR)/monte_carlo_serial.o $(BUILD_DIR)/monte_carlo_omp.o $(BUILD_DIR)/monte_carlo_opt_omp.o $(BUILD_DIR)/load_binary.o $(BUILD_DIR)/load_config.o $(BUILD_DIR)/csv_writer.o

# Target executable
TARGET = $(BIN_DIR)/monte_carlo

# Default target: compile everything into one executable
all: directories $(TARGET)
		@echo "Build complete. Run with: ./bin/monte_carlo [serial|openmp]"

# Create build directories
directories:
		@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Compile object files
$(BUILD_DIR)/monte_carlo_serial.o: $(SERIAL_SRC)
		$(CC) $(CFLAGS) $(GSL_PATH) -c $< -o $@

$(BUILD_DIR)/monte_carlo_omp.o: $(OMP_SRC)
		$(CC) $(CFLAGS) $(OMP_FLAGS) $(OMP_INCLUDE) $(GSL_PATH) -c $< -o $@

$(BUILD_DIR)/main_runner.o: $(MAIN_SRC)
		$(CC) $(CFLAGS) $(OMP_FLAGS) $(OMP_INCLUDE) $(GSL_PATH) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR)/load_binary.o: $(UTIL_SRC)
		$(CC) $(CFLAGS) $(GSL_PATH) -c $< -o $@

$(BUILD_DIR)/load_config.o: $(CONFIG_SRC)
		$(CC) $(CFLAGS) $(YAML_INCLUDE) -c $< -o $@

$(BUILD_DIR)/csv_writer.o: $(CSV_SRC)
		$(CC) $(CFLAGS) -c $< -o $@

# Link everything into single executable
$(TARGET): $(OBJS)
		$(CC) $(CFLAGS) $(LDFLAGS) $(OMP_FLAGS) $^ -o $@ $(OMP_LIBPATH) $(GSL_LIBPATH) $(YAML_LIBPATH) $(GSL_FLAGS) $(OMP_LIB) $(YAML_FLAGS)

# Clean build artifacts
clean:
		rm -rf $(BUILD_DIR) $(BIN_DIR)

# Profiling build with gprof (-pg)
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: clean directories $(TARGET)
		@echo "Profiling build complete. Run ./bin/monte_carlo to generate gmon.out"

# Clean and rebuild
rebuild: clean all

.PHONY: all clean rebuild directories profile