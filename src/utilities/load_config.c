#include "load_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yaml.h>

// Forward declarations
static void init_config_defaults(ConfigParams *config);

// Initialize config with default values
static void init_config_defaults(ConfigParams *config) {
    config->start = NULL;
    config->end = NULL;
    config->x = 0.02;
    config->N = 0;
    config->M = 100000;
    config->k = 5;
    config->random_seed = 42;
    config->train_ratio = 0.8;
    config->indices = NULL;
    config->num_indices = 0;
    config->models = NULL;
    config->num_models = 0;
    config->comment = NULL;
}

// Free all dynamically allocated memory in ConfigParams
void free_config(ConfigParams *config) {
    if (!config) return;

    if (config->start) {
        free(config->start);
        config->start = NULL;
    }
    if (config->end) {
        free(config->end);
        config->end = NULL;
    }
    if (config->indices) {
        for (int i = 0; i < config->num_indices; i++) {
            if (config->indices[i].name) free(config->indices[i].name);
            if (config->indices[i].ticker) free(config->indices[i].ticker);
        }
        free(config->indices);
        config->indices = NULL;
    }
    if (config->models) {
        for (int i = 0; i < config->num_models; i++) {
            if (config->models[i]) free(config->models[i]);
        }
        free(config->models);
        config->models = NULL;
    }
    if (config->comment) {
        free(config->comment);
        config->comment = NULL;
    }

    config->num_indices = 0;
    config->num_models = 0;
}

// Parse YAML models array
static int parse_models_array(yaml_parser_t *parser, ConfigParams *config) {
    yaml_event_t event;
    int in_sequence = 1;  // Already inside the sequence (SEQUENCE_START was consumed by caller)
    char **temp_models = NULL;
    int count = 0;

    while (1) {
        if (!yaml_parser_parse(parser, &event)) {
            fprintf(stderr, "Error: YAML parser error in models array\n");
            return -1;
        }

        switch (event.type) {
            case YAML_SEQUENCE_START_EVENT:
                in_sequence = 1;
                break;

            case YAML_SEQUENCE_END_EVENT:
                in_sequence = 0;
                yaml_event_delete(&event);
                config->models = temp_models;
                config->num_models = count;
                return 0;

            case YAML_SCALAR_EVENT:
                if (in_sequence) {
                    char *value = (char *)event.data.scalar.value;
                    char **new_models = (char **)realloc(temp_models, (count + 1) * sizeof(char *));
                    if (!new_models) {
                        fprintf(stderr, "Error: Failed to allocate memory for models\n");
                        return -1;
                    }
                    temp_models = new_models;
                    temp_models[count] = strdup(value);
                    if (!temp_models[count]) {
                        fprintf(stderr, "Error: Failed to duplicate model name\n");
                        return -1;
                    }
                    count++;
                }
                break;

            default:
                break;
        }

        yaml_event_delete(&event);
    }
}

// Parse YAML indices mapping
static int parse_indices_mapping(yaml_parser_t *parser, ConfigParams *config) {
    yaml_event_t event;
    IndexConfig *temp_indices = NULL;
    int count = 0;
    char *current_index_name = NULL;

    while (1) {
        if (!yaml_parser_parse(parser, &event)) {
            fprintf(stderr, "Error: YAML parser error in indices mapping\n");
            return -1;
        }

        switch (event.type) {
            case YAML_MAPPING_START_EVENT:
                // Start of indices mapping
                break;

            case YAML_MAPPING_END_EVENT:
                yaml_event_delete(&event);
                config->indices = temp_indices;
                config->num_indices = count;
                return 0;

            case YAML_SCALAR_EVENT: {
                char *value = (char *)event.data.scalar.value;

                if (current_index_name == NULL) {
                    // This is an index name (key)
                    current_index_name = strdup(value);
                    if (!current_index_name) {
                        fprintf(stderr, "Error: Failed to duplicate index name\n");
                        return -1;
                    }
                } else {
                    // This is the ticker value
                    IndexConfig *new_indices = (IndexConfig *)realloc(temp_indices, (count + 1) * sizeof(IndexConfig));
                    if (!new_indices) {
                        fprintf(stderr, "Error: Failed to allocate memory for indices\n");
                        return -1;
                    }
                    temp_indices = new_indices;
                    temp_indices[count].name = current_index_name;
                    temp_indices[count].ticker = strdup(value);
                    if (!temp_indices[count].ticker) {
                        fprintf(stderr, "Error: Failed to duplicate ticker\n");
                        return -1;
                    }
                    count++;
                    current_index_name = NULL;
                }
                break;
            }

            default:
                break;
        }

        yaml_event_delete(&event);
    }
}

int load_config(const char *filename, ConfigParams *config) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open config file %s\n", filename);
        return -1;
    }

    yaml_parser_t parser;
    yaml_event_t event;

    // Initialize parser
    if (!yaml_parser_initialize(&parser)) {
        fprintf(stderr, "Error: Failed to initialize YAML parser\n");
        fclose(file);
        return -1;
    }

    yaml_parser_set_input_file(&parser, file);

    // Initialize config with defaults
    init_config_defaults(config);

    char *current_key = NULL;
    int parsing = 1;

    while (parsing) {
        if (!yaml_parser_parse(&parser, &event)) {
            fprintf(stderr, "Error: YAML parser error\n");
            yaml_parser_delete(&parser);
            fclose(file);
            if (current_key) free(current_key);
            return -1;
        }

        switch (event.type) {
            case YAML_SCALAR_EVENT: {
                char *value = (char *)event.data.scalar.value;

                if (current_key == NULL) {
                    // This is a key
                    current_key = strdup(value);
                } else {
                    // This is a value
                    if (strcmp(current_key, "start") == 0) {
                        config->start = strdup(value);
                    } else if (strcmp(current_key, "end") == 0) {
                        config->end = strdup(value);
                    } else if (strcmp(current_key, "x") == 0) {
                        config->x = atof(value);
                    } else if (strcmp(current_key, "N") == 0) {
                        config->N = atoi(value);
                    } else if (strcmp(current_key, "M") == 0) {
                        config->M = atoi(value);
                    } else if (strcmp(current_key, "k") == 0) {
                        config->k = atoi(value);
                    } else if (strcmp(current_key, "random_seed") == 0) {
                        config->random_seed = (unsigned long)atol(value);
                    } else if (strcmp(current_key, "train_ratio") == 0) {
                        config->train_ratio = atof(value);
                    } else if (strcmp(current_key, "comment") == 0) {
                        config->comment = strdup(value);
                    }

                    free(current_key);
                    current_key = NULL;
                }
                break;
            }

            case YAML_MAPPING_START_EVENT:
                // Check if this is the start of indices mapping
                if (current_key && strcmp(current_key, "indices") == 0) {
                    free(current_key);
                    current_key = NULL;
                    yaml_event_delete(&event);

                    if (parse_indices_mapping(&parser, config) != 0) {
                        yaml_parser_delete(&parser);
                        fclose(file);
                        return -1;
                    }
                    continue;
                }
                break;

            case YAML_SEQUENCE_START_EVENT:
                // Check if this is the start of models array
                if (current_key && strcmp(current_key, "models") == 0) {
                    free(current_key);
                    current_key = NULL;
                    //yaml_event_delete(&event);

                    if (parse_models_array(&parser, config) != 0) {
                        yaml_parser_delete(&parser);
                        fclose(file);
                        return -1;
                    }
                    //added
                    yaml_event_delete(&event);
                    continue;
                }
                break;

            case YAML_STREAM_END_EVENT:
                parsing = 0;
                break;

            default:
                break;
        }

        yaml_event_delete(&event);
    }

    if (current_key) {
        free(current_key);
    }

    yaml_parser_delete(&parser);
    fclose(file);

    printf("Loaded configuration from %s\n", filename);
    printf("  Date range: %s to %s\n", config->start ? config->start : "N/A", config->end ? config->end : "N/A");
    printf("  M = %d (Monte Carlo trials)\n", config->M);
    printf("  x = %.4f (return threshold)\n", config->x);
    printf("  k = %d (crash threshold)\n", config->k);
    printf("  random_seed = %lu\n", config->random_seed);
    printf("  train_ratio = %.4f\n", config->train_ratio);
    printf("  Number of indices: %d\n", config->num_indices);
    printf("  Number of models: %d\n", config->num_models);
    for (int i = 0; i < config->num_models; i++) {
        printf("    - %s\n", config->models[i]);
    }
    if (config->comment) {
        printf("  Comment: %s\n", config->comment);
    }

    return 0;
}
