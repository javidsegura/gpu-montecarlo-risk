# Makefile for Monte Carlo Risk Analysis Project
# Compiles serial, OpenMP, MPI+OpenMP, and CUDA implementations

# Compiler and flags

MPI_AVAILABLE = $(shell which mpicc > /dev/null 2>&1 && echo yes || echo)
ifeq ($(MPI_AVAILABLE),yes)
    CC = mpicc
    BASE_CFLAGS = -Wall -Wextra -O3 -std=c11 -DSERIAL_BUILD -DOPENMP_BUILD -DOPENMP_OPT_BUILD -DMPI_OPENMP_BUILD
    $(info Using mpicc with MPI support enabled)
else
    # Fallback to system compiler
    CC = $(shell which clang > /dev/null 2>&1 && echo clang || echo gcc)
    BASE_CFLAGS = -Wall -Wextra -O3 -std=c11 -DSERIAL_BUILD -DOPENMP_BUILD -DOPENMP_OPT_BUILD
    $(warning MPI not available - building without MPI support)
endif


NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -std=c11 -DSERIAL_BUILD -DOPENMP_BUILD -DOPENMP_OPT_BUILD -DMPI_OPENMP_BUILD
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -arch=sm_60 -Xcompiler -fPIC
CUDA_LDFLAGS = -lcudart -lcurand

# Platform-specific library detection
LIBOMP_PATH = $(shell brew --prefix libomp 2>/dev/null)
OMP_INCLUDE = $(if $(LIBOMP_PATH),-I$(LIBOMP_PATH)/include,)
OMP_LIBPATH = $(if $(LIBOMP_PATH),-L$(LIBOMP_PATH)/lib,)
OMP_LIB = $(if $(LIBOMP_PATH),-lomp,)

GSL_PATH = $(shell pkg-config --cflags-only-I gsl 2>/dev/null || echo "-I/opt/homebrew/include")
GSL_LIBPATH = $(shell pkg-config --libs-only-L gsl 2>/dev/null || echo "-L/opt/homebrew/lib")
GSL_FLAGS = -lgsl -lgslcblas

YAML_INCLUDE = $(shell pkg-config --cflags yaml-0.1 2>/dev/null || echo "-I/opt/homebrew/include")
YAML_LIBPATH = $(shell pkg-config --libs-only-L yaml-0.1 2>/dev/null || echo "-L/opt/homebrew/lib")
YAML_FLAGS = -lyaml

# Consolidated LDFLAGS with proper library paths
LDFLAGS = $(GSL_LIBPATH) $(YAML_LIBPATH) $(OMP_LIBPATH) $(GSL_FLAGS) $(YAML_FLAGS) $(OMP_LIB) -lm -lpthread

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
OPENMP_OPT_SRC = $(SRC_DIR)/04-optimizedOpenMP/monte_carlo_opt_omp.c
MPI_OPENMP_SRC = $(SRC_DIR)/05-MPI/monte_carlo_mpi_openmp.c
CUDA_SRC = $(SRC_DIR)/06-GPU/monte_carlos_cuda.cu
MAIN_SRC = $(SRC_DIR)/main_runner.c

# Object files
UTIL_OBJ = $(OBJ_DIR)/load_binary.o $(OBJ_DIR)/load_config.o $(OBJ_DIR)/csv_writer.o
SERIAL_OBJ = $(OBJ_DIR)/monte_carlo_serial.o
OPENMP_OBJ = $(OBJ_DIR)/monte_carlo_omp.o
OPENMP_OPT_OBJ = $(OBJ_DIR)/monte_carlo_opt_omp.o
MPI_OPENMP_OBJ = $(OBJ_DIR)/monte_carlo_mpi_openmp.o
CUDA_OBJ = $(OBJ_DIR)/monte_carlo_cuda.o
MAIN_OBJ = $(OBJ_DIR)/main_runner.o

# Check if CUDA source exists and nvcc is available
CUDA_SRC_EXISTS = $(wildcard $(CUDA_SRC))
NVCC_AVAILABLE = $(shell which $(NVCC) > /dev/null 2>&1 && echo yes || echo)
CUDA_AVAILABLE = $(if $(and $(CUDA_SRC_EXISTS),$(NVCC_AVAILABLE)),yes,)

# Base object files (always needed)
BASE_OBJ = $(MAIN_OBJ) $(SERIAL_OBJ) $(OPENMP_OBJ) $(OPENMP_OPT_OBJ) $(MPI_OPENMP_OBJ) $(UTIL_OBJ)

# Conditionally include CUDA object and set linker
ifeq ($(CUDA_AVAILABLE),yes)
    ALL_OBJ = $(BASE_OBJ) $(CUDA_OBJ)
    LINKER = $(NVCC)
    LINK_FLAGS = $(NVCCFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS) -Xcompiler "$(OMPFLAGS)"
else
    ALL_OBJ = $(BASE_OBJ)
    LINKER = $(CC)
    LINK_FLAGS = $(LDFLAGS) $(OMPFLAGS)
endif

# Target executable
TARGET = $(BIN_DIR)/monte_carlo

# Default target
all: $(TARGET)
	@echo "=== Build Complete ==="
	@echo "Target: $(TARGET)"

# Link main executable - FIXED: Use conditional linker and ALL_OBJ
$(TARGET): $(ALL_OBJ)
	@echo "Linking $@..."
	$(LINKER) -o $@ $^ $(LINK_FLAGS)
	@echo "Build complete: $@"

# Compile main runner
$(MAIN_OBJ): $(MAIN_SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OMP_INCLUDE) $(GSL_PATH) $(YAML_INCLUDE) -I$(SRC_DIR) -c $< -o $@

# Compile serial model
$(SERIAL_OBJ): $(SERIAL_SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(GSL_PATH) -I$(SRC_DIR) -c $< -o $@

# Compile OpenMP model
$(OPENMP_OBJ): $(OPENMP_SRC)
	@echo "Compiling $< with OpenMP..."
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OMP_INCLUDE) $(GSL_PATH) -I$(SRC_DIR) -c $< -o $@

# Compile optimized OpenMP model - RESTORED
$(OPENMP_OPT_OBJ): $(OPENMP_OPT_SRC)
	@echo "Compiling $< with Optimized OpenMP..."
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OMP_INCLUDE) $(GSL_PATH) -I$(SRC_DIR) -c $< -o $@

# Compile MPI + OpenMP model - RESTORED
$(MPI_OPENMP_OBJ): $(MPI_OPENMP_SRC)
	@echo "Compiling $< with MPI+OpenMP..."
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OMP_INCLUDE) $(GSL_PATH) -I$(SRC_DIR) -c $< -o $@

# Compile CUDA model (conditional) - FIXED
ifeq ($(CUDA_AVAILABLE),yes)
$(CUDA_OBJ): $(CUDA_SRC)
	@echo "Compiling $< with CUDA..."
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) -c $< -o $@
endif

# Compile utilities
$(OBJ_DIR)/load_binary.o: $(UTIL_DIR)/load_binary.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(GSL_PATH) -I$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/load_config.o: $(UTIL_DIR)/load_config.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(YAML_INCLUDE) -I$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/csv_writer.o: $(UTIL_DIR)/csv_writer.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Profiling build with gprof
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: clean $(TARGET)
	@echo "Profiling build complete. Run ./bin/monte_carlo to generate gmon.out"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Clean and rebuild
rebuild: clean all

# Display build information
info:
	@echo "=== Build Configuration ==="
	@echo "C Compiler: $(CC)"
	@echo "CUDA Compiler: $(NVCC) ($(if $(CUDA_AVAILABLE),available,not available))"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "OpenMP: $(OMPFLAGS)"
	@echo "NVCCFLAGS: $(NVCCFLAGS)"
	@echo "Libraries: $(LDFLAGS)"
	@echo "Target: $(TARGET)"
	@echo "Objects: $(ALL_OBJ)"
	@echo "=========================="

# Phony targets
.PHONY: all clean rebuild profile info