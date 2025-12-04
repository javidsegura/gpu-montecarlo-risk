# Makefile for Monte Carlo Risk Analysis Project
# Compiles serial, OpenMP, and CUDA implementations

# Compiler and flags
CC = gcc
NVCC = nvcc
MPICC = mpicc
CFLAGS = -Wall -Wextra -O3 -std=c11
LDFLAGS = -lm -lgsl -lgslcblas -lyaml -lpthread
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -arch=sm_60 -Xcompiler -fPIC
CUDA_LDFLAGS = -lcudart -lcurand
MPI_LDFLAGS = -lmpi

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
OPENMP_SRC = $(SRC_DIR)/03-openMP/monte_carlo_omp.c
CUDA_SRC = $(SRC_DIR)/05-GPU/monte_carlos_cuda.cu
MAIN_SRC = $(SRC_DIR)/main_runner.c

# Object files
UTIL_OBJ = $(OBJ_DIR)/load_binary.o $(OBJ_DIR)/load_config.o $(OBJ_DIR)/csv_writer.o
SERIAL_OBJ = $(OBJ_DIR)/monte_carlo_serial.o
OPENMP_OBJ = $(OBJ_DIR)/monte_carlo_omp.o
CUDA_OBJ = $(OBJ_DIR)/monte_carlo_cuda.o
MAIN_OBJ = $(OBJ_DIR)/main_runner.o

# Check if CUDA source exists and nvcc is available (optional)
CUDA_SRC_EXISTS = $(wildcard $(CUDA_SRC))
NVCC_AVAILABLE = $(shell which $(NVCC) > /dev/null 2>&1 && echo yes || echo)
CUDA_AVAILABLE = $(if $(and $(CUDA_SRC_EXISTS),$(NVCC_AVAILABLE)),yes,)

# Check if MPI is available (optional)
MPICC_AVAILABLE = $(shell which $(MPICC) > /dev/null 2>&1 && echo yes || echo)
MPI_AVAILABLE = $(if $(MPICC_AVAILABLE),yes,)

# Base object files (always needed)
BASE_OBJ = $(MAIN_OBJ) $(SERIAL_OBJ) $(OPENMP_OBJ) $(UTIL_OBJ)

# Include CUDA object if available
ifeq ($(CUDA_AVAILABLE),)
    # CUDA not available, CPU-only build
    ALL_OBJ = $(BASE_OBJ)
    LINKER = $(CC)
    LINK_FLAGS = $(LDFLAGS) $(OMPFLAGS)
    # Add MPI support if available (for CPU-only MPI)
    ifneq ($(MPI_AVAILABLE),)
        LINKER = $(MPICC)
        LINK_FLAGS += $(MPI_LDFLAGS)
    endif
else
    # CUDA available, include it
    ALL_OBJ = $(BASE_OBJ) $(CUDA_OBJ)
    # If MPI is available, compile CUDA with MPI support
    ifneq ($(MPI_AVAILABLE),)
        # CUDA + MPI: compile with USE_MPI flag, link MPI libraries
        LINKER = $(NVCC)
        LINK_FLAGS = $(NVCCFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS) $(MPI_LDFLAGS) -Xcompiler "$(OMPFLAGS)"
    else
        # CUDA only
        LINKER = $(NVCC)
        LINK_FLAGS = $(NVCCFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS) -Xcompiler "$(OMPFLAGS)"
    endif
endif

# Target executable
TARGET = $(BIN_DIR)/monte_carlo

# Default target
all: $(TARGET)

# Link main executable
$(TARGET): $(ALL_OBJ)
	@echo "Linking $@..."
	$(LINKER) $(LINK_FLAGS) -o $@ $^
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

# Compile CUDA model (only if source exists and nvcc is available)
ifneq ($(CUDA_AVAILABLE),)
$(CUDA_OBJ): $(CUDA_SRC)
	@echo "Compiling $< with CUDA..."
ifeq ($(MPI_AVAILABLE),)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) -c $< -o $@
else
	@echo "  (with MPI support)"
	$(NVCC) $(NVCCFLAGS) -DUSE_MPI -I$(SRC_DIR) -c $< -o $@
endif
endif

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
	@echo "C Compiler: $(CC)"
	@echo "CUDA Compiler: $(NVCC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "OpenMP: $(OMPFLAGS)"
	@echo "NVCCFLAGS: $(NVCCFLAGS)"
	@echo "Libraries: $(LDFLAGS) $(CUDA_LDFLAGS)"
	@echo "Target: $(TARGET)"
	@echo "=========================="

