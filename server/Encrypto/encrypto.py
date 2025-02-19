import onnx
import argparse
import random
from pathlib import Path
import pickle

import pprint

class EncryptModel:
    def __init__(self, model, model_root_path = '../../models/', key = 'key.ek'):
        
        # basic constants
        self.key_root_path = '../../key/'
        self.keyfile = self.key_root_path + key
        self.encrypt_path = model_root_path + 'encrypt_models/'
        self.encrypt_model_filename = self.encrypt_path + Path(model).stem + '.em' # .em : encrypted model
        self.encrypt_key_filename = self.encrypt_path + Path(model).stem + '.ek' # .ek : encrypted key
        self.encrypt_table_filename = self.encrypt_path + Path(model).stem + '.pickle' # encrypted table as table
        
        # read key
        self.key = self.readKey()
        
        # get model
        self.model_name = model
        self.model_root_path = model_root_path
        self.model_path = model_root_path + model
        self.model = self.getModel()
        
        # get nodes and inits
        self.nodes = self.getNodes()
        self.inits = self.getInits()
        
        # get model input, output
        self.model_inputs = self.model.graph.input
        self.model_outputs =  self.model.graph.output
        print("model input : ", self.model_inputs)
        print("model output : ", self.model_outputs)
        
        # nodes, inits shuffle
        self.shuffleNodes()
        self.shuffleInits()
        
        # make node, init info table of model
        self.node_table = self.makeNodeInfoTable()
        self.init_table = self.makeInitInfoTable()
        
        # make node connection info of model
        self.model_info_mat = self.makeConnInfoMat()
        
        # make header
        # self.model_header = 
        
        self.saveEncryptedModel()

    def readKey(self):
        key = []
        with open(self.keyfile, 'rb') as file:
            while True:
                byte = file.read(1)
                if not byte:
                    break
                key.append(int.from_bytes(byte, byteorder='big', signed=True))
        return key

    def shuffleNodes(self):
        random.shuffle(self.nodes)

    def shuffleInits(self):
        random.shuffle(self.inits)

    def getModel(self):
        
        model = onnx.load(self.model_path)
        return model

    def getNodes(self):

        nodes = []
        for node in self.model.graph.node:
            nodes.append(node)
            
        return nodes
        
    def getInits(self):
        
        inits = []
        for init in self.model.graph.initializer:
            inits.append(init)
        
        return inits

    def getModelInfoMat(self):
        return self.model_info_mat

    def getNodeNames(self):
        return self.node_names

    def getModelName(self):
        return self.model_name

    def getModelPath(self):
        return self.model_path
    
    def getLUSize(self):
        size = 0
        for conn_info in self.model_info_mat:
            size += len(conn_info)
        
        return self.LU_size

    def findInputConnNode(self, nodes, edge_name, node_name):
        index=0
        
        for node in nodes:
            if(node.name == node_name):
                index += 1
                continue
            
            # for input in node.input:
            #     if(input == edge_name):
            #         return index
                
            for output in node.output:
                if(output == edge_name):
                    return index
            index += 1
            
        for input in self.model_inputs:
            if(input.name == edge_name):
                return index+len(self.inits)
        
        # fail
        return -1

    def findOutputConnNodes(self, nodes, edge_name, node_name):
        indices=[]
        index = 0
        
        # round nodes
        for node in nodes:
            if(node.name == node_name):
                index += 1
                continue
            
            for input in node.input:
                if(input == edge_name):
                    indices.append(index)
                
            index += 1

        # cast connection to model output
        for output in self.model_outputs:
            if(output.name == edge_name):
                indices.append(index+len(self.inits))
            index += 1

        return indices

    def findIndexFromInits(self, inits, edge_name):
        index=0
        
        for init in inits:
            if(init.name == edge_name):
                return index
            else:
                index += 1
        # fail
        return -1

    def makeConnInfoMat(self):
        mat = []
        out_err_cnt = 0
        in_err_cnt = 0
        for node in self.nodes:
            array = []
            count = 0
            for output in node.output:
                
                # indices = self.findOutputConnNodes(self.nodes, output, node.name)
                model_output_count = 0
                for model_output in self.model_outputs:
                    if(model_output.name == output):
                        array.append((count<<24) | (len(self.nodes)+len(self.inits) + model_output_count) | 0x80000000)
                    else:
                        array.append((count<<24) | 0xffffff | 0x80000000)
                    model_output_count += 1
                ## output index not required???                
                # if(len(indices) != 0):
                #     for index in indices:
                #         array.append((count<<24) | (index&0xffffff) | 0x80000000)
                #         count += 1
                    
                # else:
                #     if(out_err_cnt == 0):
                #         array.append((count<<24) | (index&0xffffff))
                #         print("last node name : ", node.name)
                #     elif(out_err_cnt > 0):
                #         print("error: output node name - ", node.name)
                #     out_err_cnt += 1
                    
                count += 1
        
            count = 0
            for input in node.input:
                
                index = self.findInputConnNode(self.nodes, input, node.name)
                if(index == -1):
                    index = self.findIndexFromInits(self.inits, input)
                    if (index != -1):
                        index += len(self.nodes)
                    
                if(index != -1):
                    array.append((count<<24) | (index&0xffffff))
                    
                else:
                    if(in_err_cnt == 0):
                        print("start node name : ", node.name)
                    elif(in_err_cnt > 0):
                        print("error: input node name - ", node.name)
                    in_err_cnt += 1
                
                count += 1
                
            mat.append(array)
        return mat

    def makeNodeInfoTable(self):
        table = []
        for node in self.nodes:
            array = []
            array.append(node.op_type)
            array.append(node.attribute)
            array.append(len(node.input))
            array.append(len(node.output))
            table.append(array)
        return table

    def makeInitInfoTable(self):
        table = []
        for init in self.inits:
            array = []
            array.append(init.data_type)
            array.append(init.dims)
            array.append(init.raw_data)
            table.append(array)
        return table

    def makeHeader(self):
        # max_rank = 8
        # header = node_num(4bytes) + init_num(4bytes) + input_num(4bytes) + output_num(4bytes) 
        #           + input shapes (input_num * 4bytes * max_rank) + output shapes (output_num * 4bytes * max_rank)
        #           + L, U, B coordinates (3 * 4bytes)
        header = []
        header.append(len(self.nodes))
        header.append(len(self.inits))
        
        input_info = self.model.graph.input
        output_info = self.model.graph.output
        
        
        

    def saveEncryptedModel(self):
        
        
        
        # model connction info save as binary
        with open(self.encrypt_model_filename, 'wb') as file:
            for array in self.model_info_mat:
                for element in array:
                    binary_data = element.to_bytes(4, byteorder='big')
                    print(binary_data)
                    file.write(binary_data)


    def makeConnInfoMat_old(self):
        mat = []
        for node in self.nodes:
            array = [0 for i in range(len(self.nodes) + len(self.inits))]
            count = 0
            for output in node.output:
                count += 1
                index = self.findInputConnNode(self.nodes, output, node.name)
                if(index == -1):
                    index = self.findIndexFromInits(self.inits, output)
                    if (index != -1):
                        index += len(self.nodes)
                
                if(index != -1):
                    array[index] = count
            
            count = 0
            for input in node.input:
                count += 1
                index = self.findInputConnNode(self.nodes, input, node.name)
                if(index == -1):
                    index = self.findIndexFromInits(self.inits, input)
                    if (index != -1):
                        index += len(self.nodes)
                    
                if(index != -1):
                    array[index] = -count
                    
            mat.append(array)
        
        for init in self.inits:
            array = [0 for i in range(len(self.nodes) + len(self.inits))]
            index = self.findInputConnNode(self.nodes, init.name, "")
            if (index != -1):
                array[index] = 1
            mat.append(array)
        
        return mat
    
    
    
def hex_format(data):
    if isinstance(data, list):
        return [hex_format(item) for item in data]
    elif isinstance(data, int):
        return hex(data)
    else:
        return data

def parsing():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--key", type=str, required=False, help="key file")
    
    return parser

def main():
    parser = parsing()
    args = parser.parse_args()
    
    if(args.key is not None):
        encrypt = EncryptModel(args.model, key=args.key)
    else:
        encrypt = EncryptModel(args.model)
    # pprint.pprint(encrypt.getModelInfoMat())
    model_info_mat = encrypt.getModelInfoMat()
    pprint.pprint(hex_format(model_info_mat))
    
    print(len(encrypt.getNodes()), len(encrypt.getInits()), len(encrypt.getNodes())+len(encrypt.getInits()))
    
    # model = getModel(args.model)
    # encrypt_model = modelEncrypt(model)
    
    # nodes = getNodes(model)
    # inits = getInits(model)
    
        
main()