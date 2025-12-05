#include "csv_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>

// Format timestamp to ISO 8601 format
static void format_timestamp(long timestamp, char *buffer, size_t buffer_size) {
    time_t t = (time_t)timestamp;
    struct tm *tm_info = localtime(&t);
    if (!tm_info) {
        fprintf(stderr, "Error: localtime failed\n");
        snprintf(buffer, buffer_size, "UNKNOWN");
        return;
    }
    strftime(buffer, buffer_size, "%Y-%m-%d %H:%M:%S", tm_info);
}

// Write CSV header
static int write_csv_header(FILE *file) {
    fprintf(file, "iterationnumber,timestamp,execution_time_ms,MC_throughput_secs,kernel_time_ms,overhead_time_ms,kernel_throughput,comment,");
    fprintf(file, "start_date,end_date,train_ratio,M,k,x,");
    fprintf(file, "model_name,seed,nodes,threads,processes,");
    fprintf(file, "indices,actual_freq,");
    fprintf(file, "P_hat,count,std_error,ci_lower,ci_upper\n");

    if (ferror(file)) {
        fprintf(stderr, "Error writing CSV header\n");
        return -1;
    }

    return 0;
}

// Prepares text data to be safely written to CSV by handling special characters (handle commas, quotes, newlines)
static char* escape_csv_field(const char *field) {
    if (!field || strlen(field) == 0) {
        char *empty = (char *)calloc(1, sizeof(char));
        if (!empty) {
            fprintf(stderr, "Error: Memory allocation failed for empty CSV field\n");
            return NULL;
        }
        return empty;
    }

    // Check if field needs escaping (contains comma, quote, or newline)
    int needs_escaping = 0;
    for (const char *p = field; *p; p++) {
        if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
            needs_escaping = 1;
            break;
        }
    }

    if (!needs_escaping) {
        return strdup(field);
    }

    // Allocate space for escaped field (worst case: all quotes doubled)
    size_t len = strlen(field);
    char *escaped = (char *)malloc(len * 2 + 3);  // +3 for quotes and null terminator
    if (!escaped) {
        return NULL;
    }

    // wrapp the entire field in quotes and double any internal quotes
    char *p = escaped;
    *p++ = '"';

    // Double quotes inside the field 
    for (const char *c = field; *c; c++) {
        if (*c == '"') {
            *p++ = '"';  
            *p++ = '"';
        } else {
            *p++ = *c;
        }
    }

    // closing quotes and null terminator
    *p++ = '"';
    *p = '\0';

    return escaped;
}

// Format indices as a string
// Returns dynamically allocated string that must be freed by caller 
// On error, prints to stderr and exits 
static char* format_indices(const IndexConfig *indices, int num_indices) {
    if (!indices || num_indices == 0) {
        char *empty = (char *)calloc(1, sizeof(char));
        if (!empty) {
            fprintf(stderr, "Error: Memory allocation failed for empty indices string\n");
            return NULL;
        }
        return empty;
    }

    // Calculate total length needed (add buffer for separators)
    size_t total_len = 0;
    for (int i = 0; i < num_indices; i++) {
        if (!indices[i].name || !indices[i].ticker) {
            fprintf(stderr, "Error: NULL index name or ticker at position %d\n", i);
            return NULL;
        }
        // Format: name:ticker; (extra bytes for ':' and ';')
         total_len += strlen(indices[i].name) + strlen(indices[i].ticker) + 2;
    }

    char *result = (char *)calloc(total_len + 1, sizeof(char));
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed for %zu bytes (indices)\n", total_len);
        return NULL;
    }

    char *p = result;
    for (int i = 0; i < num_indices; i++) {
        if (i > 0) {
            *p++ = ';'; // Separator between indices
        }
        int written = sprintf(p, "%s:%s", indices[i].name, indices[i].ticker); //name:ticker
        if (written < 0) {
            fprintf(stderr, "Error: sprintf failed at index %d\n", i);
            free(result);
            return NULL;
    }
        p += written; //"AAPL:Apple;GOOGL:Google"
    }
    *p = '\0';

    assert(strlen(result) < total_len + 1);  // Sanity check
    return result;
}

int write_results_to_csv(const char *filepath, const SimulationResultsData *data) {
    if (!filepath || !data) {
        fprintf(stderr, "Error: Invalid parameters to write_results_to_csv\n");
        return -1;
    }

    // Create results directory if it doesn't exist (thread-safe, atomic)
#ifdef _WIN32
    int mkdir_status = mkdir("results");
#else
    int mkdir_status = mkdir("results", 0755);
#endif
    if (mkdir_status == -1) {
        if (errno != EEXIST) { // Error diff to Error: Exists
            // Only fail if error is NOT "directory already exists"
            fprintf(stderr, "Error: mkdir failed: %s\n", strerror(errno));
            return -1;
        }
        // Directory exists, check if we can write to it
        if (access("results", W_OK) != 0) {
            fprintf(stderr, "Error: Cannot write to results directory: %s\n", strerror(errno));
            return -1;
        }
    }


    FILE *file = NULL;

    // Check if file exists BEFORE opening (for header-only-once guarantee)
    int file_exists_already = (access(filepath, F_OK) == 0);

    // Open in append mode; this creates file if doesn't exist
    file = fopen(filepath, "a");
    if (!file) {
        fprintf(stderr, "Error: Cannot open CSV file %s for writing: %s\n", filepath, strerror(errno));
        return -1;
    }

    // Write header if this is a NEW file (guaranteed single header by mutex)
    if (!file_exists_already) {
        if (write_csv_header(file) != 0) {
            fprintf(stderr, "Error: Failed to write CSV header\n");
            fclose(file);
            return -1;
        }
    }

    // Format timestamp
    char timestamp_str[32];
    format_timestamp(data->timestamp, timestamp_str, sizeof(timestamp_str));

    // Escape string fields (these allocate memory - freed later)
    char *comment_escaped = escape_csv_field(data->comment);
    if (!comment_escaped) {
        fprintf(stderr, "Error: Failed to escape comment field\n");
        fclose(file);
        return -1;
    }

    // Format and escape indices
    char *indices_str = format_indices(data->indices, data->num_indices);
    if (!indices_str) {
        fprintf(stderr, "Error: Failed to format indices\n");
        free(comment_escaped);
        fclose(file);
        return -1;
    }

    char *indices_escaped = escape_csv_field(indices_str);
    if (!indices_escaped) {
        fprintf(stderr, "Error: Failed to escape indices field\n");
        free(comment_escaped);
        free(indices_str);
        fclose(file);
        return -1;
    }

    // Write data row to CSV
    // if pointers null -> empty string
    int fprintf_status = fprintf(file,
            "%d,%s,%ld,%f,%.3f,%.3f,%.3f,%s,"
            "%s,%s,%.4f,%d,%d,%.4f,"
            "%s,%lu,%d,%d,%d, "
            "%s,%.6f,"
            "%.6f,%d,%.6f,%.6f,%.6f\n",
            // Metadata
            data->iteration_id,
            timestamp_str,
            data->execution_time_ms,
            data->MC_throughput_secs,
            data->kernel_time_ms,
            data->overhead_time_ms,
            data->kernel_throughput,
            comment_escaped ? comment_escaped : "",
            // Configuration
            data->start_date ? data->start_date : "",
            data->end_date ? data->end_date : "",
            data->train_ratio,
            data->M,
            data->k,
            data->x,
            // Model info
            data->model_name ? data->model_name : "",
            data->seed,
            data-> nodes,
            data->threads,
            data->processes,
            // Indices
            indices_escaped ? indices_escaped : "",
            data->actual_freq,
            // Results
            data->P_hat,
            data->count,
            data->std_error,
            data->ci_lower,
            data->ci_upper
    );

    if (fprintf_status < 0 || ferror(file)) {
        fprintf(stderr, "Error writing to CSV file: %s\n", strerror(errno));
        fclose(file);
        free(comment_escaped);
        free(indices_str);
        free(indices_escaped);
        return -1;
    }

    // Flush to ensure data is written to disk immediately
    int fflush_status = fflush(file);
    if (fflush_status != 0) {
        fprintf(stderr, "Error flushing CSV file: %s\n", strerror(errno));
        fclose(file);
        free(comment_escaped);
        free(indices_str);
        free(indices_escaped);
        return -1;
    }

    // Close file
    int fclose_status = fclose(file);
    if (fclose_status != 0) {
        fprintf(stderr, "Error closing CSV file: %s\n", strerror(errno));
        free(comment_escaped);
        free(indices_str);
        free(indices_escaped);
        return -1;
    }

    // Cleanup dynamically allocated strings
    free(comment_escaped);
    comment_escaped = NULL;
    free(indices_str);
    indices_str = NULL;
    free(indices_escaped);
    indices_escaped = NULL;


    return 0;
}
