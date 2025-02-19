#ifndef MODEL_UTILS_C
#define MODEL_UTILS_C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "model_utils.h"

Onnx__ModelProto* load_onnx_model(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open ONNX model file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);

    uint8_t *buffer = (uint8_t *)malloc(file_size);
    if (!buffer) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, file_size, file);
    fclose(file);

    Onnx__ModelProto *model = onnx__model_proto__unpack(NULL, file_size, buffer);
    free(buffer);

    if (!model) {
        fprintf(stderr, "Failed to parse ONNX model\n");
        return NULL;
    }

    if (!model->graph) {
        fprintf(stderr, "Error: model->graph is NULL\n");
        onnx__model_proto__free_unpacked(model, NULL);
        return NULL;
    }

    return model;
}

void save_onnx_model(const char *filename, Onnx__ModelProto *model) {
    size_t size = onnx__model_proto__get_packed_size(model);
    uint8_t *buffer = (uint8_t *)malloc(size);
    if (!buffer) {
        perror("Memory allocation failed");
        return;
    }

    onnx__model_proto__pack(model, buffer);

    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open output file");
        free(buffer);
        return;
    }

    fwrite(buffer, 1, size, file);
    fclose(file);
    free(buffer);
}
 
Onnx__ValueInfoProto* create_new_tensor(const char *name, Onnx__TensorProto__DataType data_type, 
                                        size_t *dims, size_t n_dims){

    Onnx__ValueInfoProto *new_tensor = malloc(sizeof(Onnx__ValueInfoProto));
    if (!new_tensor) {
        fprintf(stderr, "Memory allocation failed for new_tensor\n");
        return NULL;
    }
    onnx__value_info_proto__init(new_tensor);

    new_tensor->name = strdup(name);
    new_tensor->type = malloc(sizeof(Onnx__TypeProto));
    if (!new_tensor->type) {
        fprintf(stderr, "Memory allocation failed for type\n");
        free(new_tensor);
        return NULL;
    }
    onnx__type_proto__init(new_tensor->type);
    new_tensor->type->value_case = ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE;
    new_tensor->type->tensor_type = malloc(sizeof(Onnx__TypeProto__Tensor));
    if (!new_tensor->type->tensor_type) {
        fprintf(stderr, "Memory allocation failed for tensor_type\n");
        free(new_tensor->type);
        free(new_tensor);
        return NULL;
    }
    onnx__type_proto__tensor__init(new_tensor->type->tensor_type);
    new_tensor->type->tensor_type->elem_type = data_type;

    new_tensor->type->tensor_type->shape = malloc(sizeof(Onnx__TensorShapeProto));
    if (!new_tensor->type->tensor_type->shape) {
        fprintf(stderr, "Memory allocation failed for shape\n");
        free(new_tensor->type->tensor_type);
        free(new_tensor->type);
        free(new_tensor);
        return NULL;
    }
    onnx__tensor_shape_proto__init(new_tensor->type->tensor_type->shape);
    new_tensor->type->tensor_type->shape->n_dim = n_dims;

    new_tensor->type->tensor_type->shape->dim = malloc(n_dims * sizeof(Onnx__TensorShapeProto__Dimension *));
    if (!new_tensor->type->tensor_type->shape->dim) {
        fprintf(stderr, "Memory allocation failed for shape->dim\n");
        free(new_tensor->type->tensor_type->shape);
        free(new_tensor->type->tensor_type);
        free(new_tensor->type);
        free(new_tensor);
        return NULL;
    }

    for (size_t i = 0; i < n_dims; i++) {
        new_tensor->type->tensor_type->shape->dim[i] = malloc(sizeof(Onnx__TensorShapeProto__Dimension));
        if (!new_tensor->type->tensor_type->shape->dim[i]) {
            fprintf(stderr, "Memory allocation failed for shape->dim[%zu]\n", i);
            for (size_t j = 0; j < i; j++) {
                free(new_tensor->type->tensor_type->shape->dim[j]);
            }
            free(new_tensor->type->tensor_type->shape->dim);
            free(new_tensor->type->tensor_type->shape);
            free(new_tensor->type->tensor_type);
            free(new_tensor->type);
            free(new_tensor);
            return NULL;
        }
        onnx__tensor_shape_proto__dimension__init(new_tensor->type->tensor_type->shape->dim[i]);
        new_tensor->type->tensor_type->shape->dim[i]->value_case = ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE;
        new_tensor->type->tensor_type->shape->dim[i]->dim_value = dims[i];
    }

    return new_tensor;
}

void* free_tensor(Onnx__ValueInfoProto *tensor){
    if (!tensor) {
        return NULL;
    }

    if (tensor->name) {
        free(tensor->name);
    }

    if (tensor->type) {
        if (tensor->type->tensor_type) {
            if (tensor->type->tensor_type->shape) {
                if (tensor->type->tensor_type->shape->dim) {
                    for (size_t i = 0; i < tensor->type->tensor_type->shape->n_dim; i++) {
                        if (tensor->type->tensor_type->shape->dim[i]) {
                            free(tensor->type->tensor_type->shape->dim[i]);
                        }
                    }
                    free(tensor->type->tensor_type->shape->dim);
                }
                free(tensor->type->tensor_type->shape);
            }
            free(tensor->type->tensor_type);
        }
        free(tensor->type);
    }
    free(tensor);
    return NULL;
}

void add_new_output(Onnx__ModelProto *model, Onnx__ValueInfoProto *new_output, int index){
    if (!model || !new_output) {
        return;
    }

    if (index < 0 || index > model->graph->n_output) {
        fprintf(stderr, "Invalid index\n");
        return;
    }

    Onnx__ValueInfoProto **new_outputs = malloc((model->graph->n_output + 1) * sizeof(Onnx__ValueInfoProto *));
    if (!new_outputs) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < index; i++) {
        new_outputs[i] = model->graph->output[i];
    }

    new_outputs[index] = new_output;

    for (int i = index; i < model->graph->n_output; i++) {
        new_outputs[i + 1] = model->graph->output[i];
    }

    free(model->graph->output);
    model->graph->output = new_outputs;
    model->graph->n_output++;

    return;
}

void delete_output(Onnx__ModelProto *model, int index){
    if (!model || !model->graph) {
        return;
    }

    if (index < 0 || index >= model->graph->n_output) {
        fprintf(stderr, "Invalid index\n");
        return;
    }

    if (model->graph->n_output == 1) {
        free_tensor(model->graph->output[0]);
        free(model->graph->output);
        model->graph->output = NULL;
        model->graph->n_output = 0;
        return;
    }

    Onnx__ValueInfoProto **new_outputs = malloc((model->graph->n_output - 1) * sizeof(Onnx__ValueInfoProto *));
    if (!new_outputs) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < index; i++) {
        new_outputs[i] = model->graph->output[i];
    }

    for (int i = index + 1; i < model->graph->n_output; i++) {
        new_outputs[i - 1] = model->graph->output[i];
    }

    free_tensor(model->graph->output[index]);
    free(model->graph->output);
    model->graph->output = new_outputs;
    model->graph->n_output--;

    return;
}

void delete_output_from_string(Onnx__ModelProto *model, char* name){
    if (!model || !model->graph) {
        return;
    }

    int index = -1;
    for (int i = 0; i < model->graph->n_output; i++) {
        if (strcmp(model->graph->output[i]->name, name) == 0) {
            index = i;
            break;
        }
    }

    if (index == -1) {
        fprintf(stderr, "Output tensor with name %s not found\n", name);
        return;
    }

    return delete_output(model, index);
}

void add_new_input(Onnx__ModelProto *model, Onnx__ValueInfoProto *new_input, int index){
    if (!model || !new_input) {
        return;
    }

    if (index < 0 || index > model->graph->n_input) {
        fprintf(stderr, "Invalid index\n");
        return;
    }

    Onnx__ValueInfoProto **new_inputs = malloc((model->graph->n_input + 1) * sizeof(Onnx__ValueInfoProto *));
    if (!new_inputs) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < index; i++) {
        new_inputs[i] = model->graph->input[i];
    }

    new_inputs[index] = new_input;

    for (int i = index; i < model->graph->n_input; i++) {
        new_inputs[i + 1] = model->graph->input[i];
    }

    free(model->graph->input);
    model->graph->input = new_inputs;
    model->graph->n_input++;

    return;
}

void delete_input(Onnx__ModelProto *model, int index){
    if (!model || !model->graph) {
        return;
    }

    if (index < 0 || index >= model->graph->n_input) {
        fprintf(stderr, "Invalid index\n");
        return;
    }

    if (model->graph->n_input == 1) {
        free_tensor(model->graph->input[0]);
        free(model->graph->input);
        model->graph->input = NULL;
        model->graph->n_input = 0;
        return;
    }

    Onnx__ValueInfoProto **new_inputs = malloc((model->graph->n_input - 1) * sizeof(Onnx__ValueInfoProto *));
    if (!new_inputs) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < index; i++) {
        new_inputs[i] = model->graph->input[i];
    }

    for (int i = index + 1; i < model->graph->n_input; i++) {
        new_inputs[i - 1] = model->graph->input[i];
    }

    free_tensor(model->graph->input[index]);
    free(model->graph->input);
    model->graph->input = new_inputs;
    model->graph->n_input--;

    return;
}

void delete_input_from_string(Onnx__ModelProto *model, char* name){
    if (!model || !model->graph) {
        return;
    }

    int index = -1;
    for (int i = 0; i < model->graph->n_input; i++) {
        if (strcmp(model->graph->input[i]->name, name) == 0) {
            index = i;
            break;
        }
    }

    if (index == -1) {
        fprintf(stderr, "Input tensor with name %s not found\n", name);
        return;
    }
    delete_input(model, index);
    return;
}

Onnx__AttributeProto** create_new_attribute(const char **names, Onnx__AttributeProto__AttributeType *types, void **values, size_t n_attributes){
    Onnx__AttributeProto **new_attributes = malloc(n_attributes * sizeof(Onnx__AttributeProto *));
    if (!new_attributes) {
        perror("Memory allocation failed");
        return NULL;
    }

    for (size_t i = 0; i < n_attributes; i++) {
        new_attributes[i] = malloc(sizeof(Onnx__AttributeProto));
        if (!new_attributes[i] || !names[i] || !values[i]) {
            perror("Memory allocation failed");
            for (size_t j = 0; j < i; j++) {
                free(new_attributes[j]);
            }
            free(new_attributes);
            return NULL;
        }
        onnx__attribute_proto__init(new_attributes[i]);
        new_attributes[i]->name = strdup(names[i]);
        new_attributes[i]->type = types[i];
        switch (types[i]) {
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
                new_attributes[i]->f = *(float *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
                new_attributes[i]->i = *(int64_t *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
                new_attributes[i]->s.data = (uint8_t *)strdup((char *)values[i]);
                new_attributes[i]->s.len = strlen((char *)values[i]);
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
                new_attributes[i]->t = (Onnx__TensorProto *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH:
                new_attributes[i]->g = (Onnx__GraphProto *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR:
                new_attributes[i]->sparse_tensor = (Onnx__SparseTensorProto *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
                new_attributes[i]->floats = (ProtobufCBinaryData *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
                new_attributes[i]->ints = (ProtobufCBinaryData *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
                new_attributes[i]->strings = (ProtobufCBinaryData *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSORS:
                new_attributes[i]->tensors = (ProtobufCBinaryData *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPHS:
                new_attributes[i]->graphs = (ProtobufCBinaryData *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSORS:
                new_attributes[i]->sparse_tensors = (ProtobufCBinaryData *)values[i];
                break;
            case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TYPE_PROTOS:
                new_attributes[i]->type_protos = (ProtobufCBinaryData *)values[i];
                break;
            default:
                fprintf(stderr, "Unsupported attribute type\n");
                for (size_t j = 0; j < i; j++) {
                    free(new_attributes[j]);
                }
                free(new_attributes);
                return NULL;
        }
    }
    return new_attributes;
}


Onnx__NodeProto* create_new_node(const char *name, const char *op_type, const char *domain, const char *overload, 
                                Onnx__AttributeProto **attribute, size_t n_attribute, const char *doc_string){

    Onnx__NodeProto *new_node = malloc(sizeof(Onnx__NodeProto));
    if (!new_node) {
        fprintf(stderr, "Memory allocation failed for new_node\n");
        return NULL;
    }
    onnx__node_proto__init(new_node);

        onnx__node_proto__init(new_node);

    new_node->name = strdup(name);
    if (!new_node->name) {
        fprintf(stderr, "Memory allocation failed for new_node->name\n");
        free(new_node);
        return NULL;
    }

    new_node->op_type = strdup(op_type);
    if (!new_node->op_type) {
        fprintf(stderr, "Memory allocation failed for new_node->op_type\n");
        free(new_node->name);
        free(new_node);
        return NULL;
    }

    new_node->domain = strdup(domain);
    if (!new_node->domain) {
        fprintf(stderr, "Memory allocation failed for new_node->domain\n");
        free(new_node->op_type);
        free(new_node->name);
        free(new_node);
        return NULL;
    }

    if (overload) {
        new_node->overload = strdup(overload);
        if (!new_node->overload) {
            fprintf(stderr, "Memory allocation failed for new_node->overload\n");
            free(new_node->domain);
            free(new_node->op_type);
            free(new_node->name);
            free(new_node);
            return NULL;
        }
    } else {
        new_node->overload = NULL;
    }

    new_node->attribute = attribute;
    new_node->n_attribute = n_attribute;

    if (doc_string) {
        new_node->doc_string = strdup(doc_string);
        if (!new_node->doc_string) {
            fprintf(stderr, "Memory allocation failed for new_node->doc_string\n");
            free(new_node->overload);
            free(new_node->domain);
            free(new_node->op_type);
            free(new_node->name);
            free(new_node);
            return NULL;
        }
    } else {
        new_node->doc_string = NULL;
    }

    return new_node;

}

void add_new_node(Onnx__ModelProto *model, Onnx__NodeProto *new_node, int index){
    if (!model || !new_node) {
        return;
    }

    if (index < 0 || index > model->graph->n_node) {
        fprintf(stderr, "Invalid index\n");
        return;
    }

    Onnx__NodeProto **new_nodes = malloc((model->graph->n_node + 1) * sizeof(Onnx__NodeProto *));
    if (!new_nodes) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < index; i++) {
        new_nodes[i] = model->graph->node[i];
    }

    new_nodes[index] = new_node;

    for (int i = index; i < model->graph->n_node; i++) {
        new_nodes[i + 1] = model->graph->node[i];
    }

    free(model->graph->node);
    model->graph->node = new_nodes;
    model->graph->n_node++;

    return;
}

void delete_node(Onnx__ModelProto *model, int index){
    if (!model || !model->graph) {
        return;
    }

    if (index < 0 || index >= model->graph->n_node) {
        fprintf(stderr, "Invalid index\n");
        return;
    }

    if (model->graph->n_node == 1) {
        free(model->graph->node[0]);
        free(model->graph->node);
        model->graph->node = NULL;
        model->graph->n_node = 0;
        return;
    }

    Onnx__NodeProto **new_nodes = malloc((model->graph->n_node - 1) * sizeof(Onnx__NodeProto *));
    if (!new_nodes) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < index; i++) {
        new_nodes[i] = model->graph->node[i];
    }

    for (int i = index + 1; i < model->graph->n_node; i++) {
        new_nodes[i - 1] = model->graph->node[i];
    }

    free(model->graph->node[index]);
    free(model->graph->node);
    model->graph->node = new_nodes;
    model->graph->n_node--;

    return;
}

void delete_node_from_string(Onnx__ModelProto *model, char* name){
    if (!model || !model->graph) {
        return;
    }

    int index = -1;
    for (int i = 0; i < model->graph->n_node; i++) {
        if (strcmp(model->graph->node[i]->name, name) == 0) {
            index = i;
            break;
        }
    }

    if (index == -1) {
        fprintf(stderr, "Node with name %s not found\n", name);
        return;
    }
    delete_node(model, index);
    return;
}

Onnx__TensorProto* create_new_initializer(const char *name, Onnx__TensorProto__DataType data_type, 
                                        size_t *dims, size_t n_dims, void *raw_data, size_t n_raw_data){
    Onnx__TensorProto *new_initializer = malloc(sizeof(Onnx__TensorProto));
    if (!new_initializer) {
        fprintf(stderr, "Memory allocation failed for new_initializer\n");
        return NULL;
    }
    onnx__tensor_proto__init(new_initializer);

    new_initializer->name = strdup(name);
    if (!new_initializer->name) {
        fprintf(stderr, "Memory allocation failed for new_initializer->name\n");
        free(new_initializer);
        return NULL;
    }

    new_initializer->data_type = data_type;

    new_initializer->dims = malloc(n_dims * sizeof(size_t));
    if (!new_initializer->dims) {
        fprintf(stderr, "Memory allocation failed for new_initializer->dims\n");
        free(new_initializer->name);
        free(new_initializer);
        return NULL;
    }

    for (size_t i = 0; i < n_dims; i++) {
        new_initializer->dims[i] = dims[i];
    }
    new_initializer->n_dims = n_dims;

    new_initializer->raw_data.data = malloc(n_raw_data);
    if (!new_initializer->raw_data.data) {
        fprintf(stderr, "Memory allocation failed for new_initializer->raw_data.data\n");
        free(new_initializer->dims);
        free(new_initializer->name);
        free(new_initializer);
        return NULL;
    }

    memcpy(new_initializer->raw_data.data, raw_data, n_raw_data);
    new_initializer->raw_data.len = n_raw_data;

    return new_initializer;
}

void add_new_initializer(Onnx__ModelProto *model, Onnx__TensorProto *new_initializer, int index){
    if (!model || !new_initializer) {
        return;
    }

    if (index < 0 || index > model->graph->n_initializer) {
        fprintf(stderr, "Invalid index\n");
        return;
    }

    Onnx__TensorProto **new_initializers = malloc((model->graph->n_initializer + 1) * sizeof(Onnx__TensorProto *));
    if (!new_initializers) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < index; i++) {
        new_initializers[i] = model->graph->initializer[i];
    }

    new_initializers[index] = new_initializer;

    for (int i = index; i < model->graph->n_initializer; i++) {
        new_initializers[i + 1] = model->graph->initializer[i];
    }

    free(model->graph->initializer);
    model->graph->initializer = new_initializers;
    model->graph->n_initializer++;

    return;
}

int64_t get_node_size(Onnx__ModelProto *model){
    if (!model || !model->graph) {
        return -1;
    }

    return model->graph->n_node;
}

int64_t get_node_index(Onnx__ModelProto *model, char* node_name){
    if (!model || !model->graph) {
        return -1;
    }

    for (int i = 0; i < model->graph->n_node; i++) {
        if (strcmp(model->graph->node[i]->name, node_name) == 0) {
            return i;
        }
    }

    return -1;
}

int64_t get_model_input_size(Onnx__ModelProto *model){
    if (!model || !model->graph) {
        return -1;
    }

    return model->graph->n_input;
}

int64_t get_model_input_index(Onnx__ModelProto *model, char* input_name){
    if (!model || !model->graph) {
        return -1;
    }

    for (int i = 0; i < model->graph->n_input; i++) {
        if (strcmp(model->graph->input[i]->name, input_name) == 0) {
            return i;
        }
    }

    return -1;
}

int64_t get_model_output_size(Onnx__ModelProto *model){
    if (!model || !model->graph) {
        return -1;
    }

    return model->graph->n_output;
}

int64_t get_model_output_index(Onnx__ModelProto *model, char* output_name){
    if (!model || !model->graph) {
        return -1;
    }

    for (int i = 0; i < model->graph->n_output; i++) {
        if (strcmp(model->graph->output[i]->name, output_name) == 0) {
            return i;
        }
    }

    return -1;
}

int64_t get_initializer_size(Onnx__ModelProto *model){
    if (!model || !model->graph) {
        return -1;
    }

    return model->graph->n_initializer;
}

int64_t get_model_initializer_index(Onnx__ModelProto *model, char* initializer_name){
    if (!model || !model->graph) {
        return -1;
    }

    for (int i = 0; i < model->graph->n_initializer; i++) {
        if (strcmp(model->graph->initializer[i]->name, initializer_name) == 0) {
            return i;
        }
    }

    return -1;
}

char** get_node_inputs(Onnx__ModelProto *model, char* node_name){
    if (!model || !model->graph) {
        return NULL;
    }

    int index = get_node_index(model, node_name);
    if (index == -1) {
        return NULL;
    }

    Onnx__NodeProto *node = model->graph->node[index];
    if (!node) {
        return NULL;
    }

    char **inputs = malloc(node->n_input * sizeof(char *));
    if (!inputs) {
        perror("Memory allocation failed");
        return NULL;
    }

    for (size_t i = 0; i < node->n_input; i++) {
        inputs[i] = strdup(node->input[i]);
        if (!inputs[i]) {
            perror("Memory allocation failed");
            for (size_t j = 0; j < i; j++) {
                free(inputs[j]);
            }
            free(inputs);
            return NULL;
        }
    }

    return inputs;
}

char** get_node_outputs(Onnx__ModelProto *model, char* node_name){
    if (!model || !model->graph) {
        return NULL;
    }

    int index = get_node_index(model, node_name);
    if (index == -1) {
        return NULL;
    }

    Onnx__NodeProto *node = model->graph->node[index];
    if (!node) {
        return NULL;
    }

    char **outputs = malloc(node->n_output * sizeof(char *));
    if (!outputs) {
        perror("Memory allocation failed");
        return NULL;
    }

    for (size_t i = 0; i < node->n_output; i++) {
        outputs[i] = strdup(node->output[i]);
        if (!outputs[i]) {
            perror("Memory allocation failed");
            for (size_t j = 0; j < i; j++) {
                free(outputs[j]);
            }
            free(outputs);
            return NULL;
        }
    }

    return outputs;
}


#endif  // MODEL_UTILS_C