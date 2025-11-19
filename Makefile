# Makefile for Monte Carlo Risk Analysis Project
# Compiles serial, OpenMP, and CUDA implementations

# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -std=c11
LDFLAGS = -lm -lgsl -lgslcblas -lyaml -lpthread
OMPFLAGS = -fopenmp

# Directories
SRC_DIR = src
UTIL_DIR = $(SRC_DIR)/utilities
BIN_DIR = bin
OBJ_DIR = obj

# Create directories if they don't exist
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

# Source files
UTIL_SRC = $(UTIL_DIR)/load_binary.c $(UTIL_DIR)/load_config.c $(UTIL_DIR)/csv_writer.c
SERIAL_SRC = $(SRC_DIR)/02-C-serial/monte_carlo_serial.c
OPENMP_SRC = $(SRC_DIR)/02-openMP/monte_carlo_omp.c
MAIN_SRC = $(SRC_DIR)/main_runner.c

# Object files
UTIL_OBJ = $(OBJ_DIR)/load_binary.o $(OBJ_DIR)/load_config.o $(OBJ_DIR)/csv_writer.o
SERIAL_OBJ = $(OBJ_DIR)/monte_carlo_serial.o
OPENMP_OBJ = $(OBJ_DIR)/monte_carlo_omp.o
MAIN_OBJ = $(OBJ_DIR)/main_runner.o

# Target executable
TARGET = $(BIN_DIR)/monte_carlo

# Default target
all: $(TARGET)

# Link main executable
$(TARGET): $(MAIN_OBJ) $(SERIAL_OBJ) $(OPENMP_OBJ) $(UTIL_OBJ)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"

# Compile main runner
$(MAIN_OBJ): $(MAIN_SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Compile serial model
$(SERIAL_OBJ): $(SERIAL_SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Compile OpenMP model
$(OPENMP_OBJ): $(OPENMP_SRC)
	@echo "Compiling $< with OpenMP..."
	$(CC) $(CFLAGS) $(OMPFLAGS) -I$(SRC_DIR) -c $< -o $@

# Compile utilities
$(OBJ_DIR)/load_binary.o: $(UTIL_DIR)/load_binary.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/load_config.o: $(UTIL_DIR)/load_config.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/csv_writer.o: $(UTIL_DIR)/csv_writer.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Clean and rebuild
rebuild: clean all

# Phony targets
.PHONY: all clean rebuild

# Display build information
info:
	@echo "=== Build Configuration ==="
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "OpenMP: $(OMPFLAGS)"
	@echo "Libraries: $(LDFLAGS)"
	@echo "Target: $(TARGET)"
	@echo "=========================="

