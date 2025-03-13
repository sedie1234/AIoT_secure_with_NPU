#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"

/*** Load Model ***/
/*
    * Load ONNX model from file
    * @param filename: path to the ONNX model file
    * @return Onnx__ModelProto: pointer to the loaded ONNX model
    * @return NULL: if failed to load the model
    * Usage : Onnx__ModelProto *model = load_onnx_model("model.onnx");
 */
Onnx__ModelProto* load_onnx_model(const char *filename);

/*** Save Model ***/
/*
    * Save ONNX model to file
    * @param filename: path to the output ONNX model file
    * @param model: pointer to the ONNX model to save
    * Usage : save_onnx_model("model.onnx", model);
 */
void save_onnx_model(const char *filename, Onnx__ModelProto *model);

/*** Create New Tensor ***/
/*
    * Create a new tensor
    * @param name: name of the tensor
    * @param data_type: data type of the tensor
    * @param dims: array of dimensions of the tensor
    * @param n_dims: number of dimensions
    * @return Onnx__ValueInfoProto: pointer to the created tensor
    * Usage : Onnx__ValueInfoProto *new_tensor = create_new_tensor("input", ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT, dims, n_dims);
*/
Onnx__ValueInfoProto* create_new_tensor(const char *name, Onnx__TensorProto__DataType data_type, 
                                        size_t *dims, size_t n_dims);

/*** Free Tensor ***/
/*
    * Free the memory allocated for a tensor\
    * if the tensor used in the model, it should be removed from the model before freeing
    * or the tensor automatically freed when the model is freed
    * @param tensor: pointer to the tensor to free
    * @return void
    * Usage : free_tensor(tensor);
*/
void* free_tensor(Onnx__ValueInfoProto *tensor);

/*** Add New Output ***/
/*
    * Add a new output to the model
    * @param model: pointer to the ONNX model
    * @param new_output: pointer to the new output tensor
    * @param index: index of the new output tensor, 0 if the model has no output tensors
    * @return void
    * Usage : add_new_output(model, new_output, 0);
*/
void add_new_output(Onnx__ModelProto *model, Onnx__ValueInfoProto *new_output, int index);

/*** Delete Output ***/
/*
    * Delete an output from the model
    * @param model: pointer to the ONNX model
    * @param index: index of the output tensor to delete
    * @return void
    * Usage : delete_output(model, 0);
*/
void delete_output(Onnx__ModelProto *model, int index);

/*** Delete Output From String ***/
/*
    * Delete an output from the model by name
    * @param model: pointer to the ONNX model
    * @param name: name of the output tensor to delete
    * @return void
    * Usage : delete_output_from_string(model, "output");
*/
void delete_output_from_string(Onnx__ModelProto *model, char* name);

/*** Add New Input ***/
/*
    * Add a new input to the model
    * @param model: pointer to the ONNX model
    * @param new_input: pointer to the new input tensor
    * @param index: index of the new input tensor, 0 if the model has no input tensors
    * @return void
    * Usage : add_new_input(model, new_input, 0);
*/
void add_new_input(Onnx__ModelProto *model, Onnx__ValueInfoProto *new_input, int index);

/*** Delete Input ***/
/*
    * Delete an input from the model
    * @param model: pointer to the ONNX model
    * @param index: index of the input tensor to delete
    * @return void
    * Usage : delete_input(model, 0);
*/
void delete_input(Onnx__ModelProto *model, int index);

/*** Delete Input From String ***/
/*
    * Delete an input from the model by name
    * @param model: pointer to the ONNX model
    * @param name: name of the input tensor to delete
    * @return void
    * Usage : delete_input_from_string(model, "input");
*/
void delete_input_from_string(Onnx__ModelProto *model, char* name);

/*** Create New Attribute ***/
/*
    * Create a new attribute
    * @param name: name of the attribute
    * @param type: type of the attribute
    * @param value: value of the attribute
    * @return Onnx__AttributeProto: pointer to the created attribute
    * Usage : Onnx__AttributeProto *new_attribute = create_new_attribute("attr", ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT, &value);
*/
Onnx__AttributeProto** create_new_attribute(const char **names, Onnx__AttributeProto__AttributeType *types, void **values, size_t n_attributes);

/*** Get Attribute ***/
/*
    * Get the attribute of a node by name
    * @param node: pointer to the node
    * @return Onnx__AttributeProto: pointer to the attribute, NULL if the attribute does not exist
    * Usage : Onnx__AttributeProto **attr = get_attribute(node);
*/
Onnx__AttributeProto* get_attribute(Onnx__NodeProto *node);

/*** Get Attributes ***/
/*
    * Get the attributes of a model
    * @param model: pointer to the ONNX model
    * @return Onnx__AttributeProto: pointer to the attributes
    * Usage : Onnx__AttributeProto **attributes = get_attributes(model);
*/
Onnx__AttributeProto** get_attributes(Onnx__ModelProto *model);

/*** Free Attribute ***/
/*
    * Free the memory allocated for an attribute
    * if the attribute used in the model, it should be removed from the model before freeing
    * or the attribute automatically freed when the model is freed
    * @param attribute: pointer to the attribute to free
    * @return void
    * Usage : free_attribute(attribute);
*/
void* free_attribute(Onnx__AttributeProto *attribute);

/*** Free Attributes ***/
/*
    * Free the memory allocated for an array of attributes
    * if the attributes used in the model, it should be removed from the model before freeing
    * or the attributes automatically freed when the model is freed
    * @param attributes: pointer to the attributes to free
    * @return void
    * Usage : free_attributes(attributes);
*/
void* free_attributes(Onnx__AttributeProto **attributes, size_t n_attributes);

/*** Create New Node ***/
/*
    * Create a new node
    * @param name: name of the node
    * @param op_type: type of the operator to execute
    * @param domain: domain of the operator set that specifies the operator
    * @param overload: overload identifier to map this to a model-local function
    * @param attribute: array of additional named attributes
    * @param n_attribute: number of attributes
    * @param doc_string: human-readable documentation for the node
    * @return Onnx__NodeProto: pointer to the created node
    * Usage : Onnx__NodeProto *new_node = create_new_node("node", "Conv", "ai.onnx", NULL, attributes, n_attributes, "Convolutional layer");
*/
Onnx__NodeProto* create_new_node(const char *name, const char *op_type, const char *domain, const char *overload, 
                                Onnx__AttributeProto **attribute, size_t n_attribute, const char *doc_string);

/*** Add New Node ***/
/*
    * Add a new node to the model
    * @param model: pointer to the ONNX model
    * @param new_node: pointer to the new node
    * @param index: index of the new node, 0 if the model has no nodes
    * @return void
    * Usage : add_new_node(model, new_node, 0);
*/
void add_new_node(Onnx__ModelProto *model, Onnx__NodeProto *new_node, int index);

/*** Delete Node ***/
/*
    * Delete a node from the model
    * @param model: pointer to the ONNX model
    * @param index: index of the node to delete
    * @return void
    * Usage : delete_node(model, 0);
*/
void delete_node(Onnx__ModelProto *model, int index);

/*** Delete Node From String ***/
/*
    * Delete a node from the model by name
    * @param model: pointer to the ONNX model
    * @param name: name of the node to delete
    * @return void
    * Usage : delete_node_from_string(model, "node");
*/
void delete_node_from_string(Onnx__ModelProto *model, char* name);

/*** Create New Initializer ***/
/*
    * Create a new initializer
    * @param name: name of the initializer
    * @param data_type: data type of the initializer
    * @param dims: array of dimensions of the initializer
    * @param n_dims: number of dimensions
    * @param raw_data: raw data of the initializer
    * @param n_raw_data: number of raw data elements
    * @return Onnx__TensorProto: pointer to the created initializer
    * Usage : Onnx__TensorProto *new_initializer = create_new_initializer("weight", ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT, dims, n_dims, raw_data, n_raw_data);
*/
Onnx__TensorProto* create_new_initializer(const char *name, Onnx__TensorProto__DataType data_type, 
                                        size_t *dims, size_t n_dims, void *raw_data, size_t n_raw_data);

/*** Add New Initializer ***/
/*
    * Add a new initializer to the model
    * @param model: pointer to the ONNX model
    * @param new_initializer: pointer to the new initializer
    * @param index: index of the new initializer, 0 if the model has no initializers
    * @return void
    * Usage : void add_new_initializer(model, new_initializer, 0);
*/
void add_new_initializer(Onnx__ModelProto *model, Onnx__TensorProto *new_initializer, int index);

/*** Get # of Node ***/
/*
    * Get the number of nodes in the model
    * @param model: pointer to the ONNX model
    * @return int64_t: number of nodes in the model
    * Usage : int64_t n_nodes = get_node_size(model);
*/
int64_t get_node_size(Onnx__ModelProto *model);

/*** Get Node Index ***/
/*
    * Get the index of a node in the model
    * @param model: pointer to the ONNX model
    * @param node_name: name of the node
    * @return int64_t: index of the node in the model, -1 if the node does not exist
    * Usage : int64_t index = get_node_index(model, "node");
*/
int64_t get_node_index(Onnx__ModelProto *model, char* node_name);

/*** Get # of Model Inputs ***/
/*
    * Get the number of input tensors in the model
    * @param model: pointer to the ONNX model
    * @return int64_t: number of input tensors in the model
    * Usage : int64_t n_inputs = get_model_input_size(model);
*/
int64_t get_model_input_size(Onnx__ModelProto *model);

/*** Get Model Input Index ***/
/*
    * Get the index of an input tensor in the model
    * @param model: pointer to the ONNX model
    * @param input_name: name of the input tensor
    * @return int64_t: index of the input tensor in the model, -1 if the input tensor does not exist
    * Usage : int64_t index = get_model_input_index(model, "input");
*/
int64_t get_model_input_index(Onnx__ModelProto *model, char* input_name);

/*** Get # of Model Outputs ***/
/*
    * Get the number of output tensors in the model
    * @param model: pointer to the ONNX model
    * @return int64_t: number of output tensors in the model
    * Usage : int64_t n_outputs = get_model_output_size(model);
*/
int64_t get_model_output_size(Onnx__ModelProto *model);

/*** Get Model Output Index ***/
/*
    * Get the index of an output tensor in the model
    * @param model: pointer to the ONNX model
    * @param output_name: name of the output tensor
    * @return int64_t: index of the output tensor in the model, -1 if the output tensor does not exist
    * Usage : int64_t index = get_model_output_index(model, "output");
*/
int64_t get_model_output_index(Onnx__ModelProto *model, char* output_name);

/*** Get # of Initializers ***/
/*
    * Get the number of initializers in the model
    * @param model: pointer to the ONNX model
    * @return int64_t: number of initializers in the model
    * Usage : int64_t n_initializers = get_initializer_size(model);
*/
int64_t get_initializer_size(Onnx__ModelProto *model);

/*** Get Initializer Index ***/
/*
    * Get the index of an initializer in the model
    * @param model: pointer to the ONNX model
    * @param initializer_name: name of the initializer
    * @return int64_t: index of the initializer in the model, -1 if the initializer does not exist
    * Usage : int64_t index = get_initializer_index(model, "weight");
*/
int64_t get_initializer_index(Onnx__ModelProto *model, char* initializer_name);

/*** Get Node Inputs ***/
/*
    * Get the names of the input tensors of a node in the model
    * @param model: pointer to the ONNX model
    * @param node_name: name of the node
    * @return char**: array of input tensor names, NULL if the node does not exist
    * Usage : char** inputs = get_node_inputs(model, "node");
*/
char** get_node_inputs(Onnx__ModelProto *model, char* node_name);

/*** Get Node Outputs ***/
/*
    * Get the names of the output tensors of a node in the model
    * @param model: pointer to the ONNX model
    * @param node_name: name of the node
    * @return char**: array of output tensor names, NULL if the node does not exist
    * Usage : char** outputs = get_node_outputs(model, "node");
*/
char** get_node_outputs(Onnx__ModelProto *model, char* node_name);



#endif // MODEL_UTILS_H