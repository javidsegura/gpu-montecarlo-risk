#include "load_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yaml.h>

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
    config->M = 100000;
    config->x = 0.02;
    config->k = 5;

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
            case YAML_SCALAR_EVENT:
                if (current_key == NULL) {
                    // This is a key
                    current_key = strdup((char *)event.data.scalar.value);
                } else {
                    // This is a value
                    char *value = (char *)event.data.scalar.value;

                    if (strcmp(current_key, "M") == 0) {
                        config->M = atoi(value);
                    } else if (strcmp(current_key, "x") == 0) {
                        config->x = atof(value);
                    } else if (strcmp(current_key, "k") == 0) {
                        config->k = atoi(value);
                    }

                    free(current_key);
                    current_key = NULL;
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
    printf("  M = %d\n", config->M);
    printf("  x = %.4f\n", config->x);
    printf("  k = %d\n", config->k);

    return 0;
}
