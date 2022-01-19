 # -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:33:11 2021

@author: 12037
"""

import random
import numpy as np
import copy
import os
import sys
import matplotlib.pyplot as plt 

experiment_weightxcost_log = []

cia_init_flag = 0

class logging():
    def new_section():
        print('                                                ')
    
    def barred(text):
        print('------------------------------------------------')
        print(text)
        print('------------------------------------------------')
        
    def below_barred(text):
        print(text)
        print('------------------------------------------------')
        
    def bold_barred(text):
        print('================================================')
        print(text)
        print('================================================')
        
    def below_bold_barred(text):
        print(text)
        print('================================================')

#CIA: Calculation integrity assurance 
class cia_archiver():
    def log_layeroutput(layer, layer_output):
        if (cia_init_flag == 1):
            return 0
        
        logger = open('C:/Users/12037/.spyder-py3/cognitare/CIA_log.txt', 'a')
        logger.write('LAYER '+str(layer)+' - OUTPUT: '+str(layer_output)+'\n')
        logger.write('-------------------------------------------------------------------\n')    
    
    def log_newlayer(layer):
        if (cia_init_flag == 1):
            return 0
        
        logger = open('C:/Users/12037/.spyder-py3/cognitare/CIA_log.txt', 'a')
        logger.write('-------------------------- CALCULATING LAYER '+str(layer)+' --------------------------\n')
    
    def log_neuronoutput(neuron, weight_det, neuron_input, weight, bias, output):
        if (cia_init_flag == 1):
            return 0
        
        logger = open('C:/Users/12037/.spyder-py3/cognitare/CIA_log.txt', 'a')
        logger.write('CALCULATING NEURON '+str(neuron)+' - WEIGHT: '+str(weight_det)+' '+str(neuron_input)+'*'+str(weight)+'+'+str(bias)+'='+str(output)+'\n')
        logger.write('-------------------------------------------------------------------\n')

    def log_activation(activation):
        if (cia_init_flag == 1):
            return 0
        
        logger = open('C:/Users/12037/.spyder-py3/cognitare/CIA_log.txt', 'a')
        logger.write('ACTIVATION TOTAL: '+str(activation)+'\n')
        logger.write('-------------------------------------------------------------------\n')


cia_archiver.init_flag = 1

class check():
    def layers():
        error_flag = 0
        
        x=0
        while (x < len(model.metadata.layers)):
            if (model.metadata.layers[x].identifier != 'COGNITARE LAYER'):
                logging.below_barred('LAYER '+str(x)+': '+str(model.metadata.layers[x]) + ' - ERROR INCOMPATIBLE LAYER')
                error_flag = 1
            else:
                logging.below_barred('LAYER '+str(x)+': '+str(model.metadata.layers[x]))
            x=x+1
        
        if (error_flag == 1):
            return False
        else:
            return True

class net_manage():        
    def generate_bias(model):        
        x=0
        while (x < len(model.metadata.layers)):
            y=0
            while (y < model.metadata.layers[x].neuron_amount):
                rand_neuron = random.random()
                model.metadata.layers[x].neuron_bias.append(rand_neuron)
                y=y+1
            x=x+1
        logging.below_barred('Bias generation complete')

    def generate_weight(model):
        x=0
        # For each layer in model
        while (x < len(model.metadata.layers)):
            if (x < len(model.metadata.layers)-1):
            #If layer is a input or hidden layer
                y=0
                #For each neuron in layer
                while (y < model.metadata.layers[x].neuron_amount):
                    neural_weights = []
                    z=0
                    #For each neuron in next layer
                    while (z < model.metadata.layers[x+1].neuron_amount):
                        rand_connection = random.random()
                        neural_weights.append(rand_connection)
                        z=z+1    
                    model.metadata.layers[x].neuron_weights.append(neural_weights)
                    y=y+1  
            x=x+1
        
        logging.below_barred('Weight generation complete')

class loss():
    class regression():
        def MSE(predictions, targets):
            numpy_predictions = np.array(predictions)
            numpy_targets = np.array(targets)
            
            differences = numpy_predictions - numpy_targets
            differences_squared = differences ** 2
            mean_of_differences_squared = differences_squared.mean()
            logging.below_barred('LOSS: '+mean_of_differences_squared)
            return mean_of_differences_squared
        
        def RMSE(predictions, targets):
            logging.below_barred('RMSE - Predictions Input: '+str(predictions))
            logging.below_barred('RMSE - Targets Input: '+str(targets))
            
            #plt.title('Predictions')
            #plt.plot(predictions)
            #plt.show()
            
            #plt.title('Targets')
            #plt.plot(targets)
            #plt.show()
    
            numpy_predictions = np.array(predictions)
            numpy_targets = np.array(targets)
            
            differences = numpy_predictions - numpy_targets
            differences_squared = differences ** 2
            mean_of_differences_squared = differences_squared.mean()
            return (np.sqrt(mean_of_differences_squared), np.sqrt(differences_squared))
            
        
        
        def CE(predictions, targets):
            return 0

class activation():
    def standard(inputs, layer): #This is linear
        layer_activation = []
        
        x=0
        while (x < len(layer.neuron_weights[0])): 
            y=0
            while (y < len(layer.neuron_weights)):
                activation=0
                activation += layer.neuron_weights[y][x] * inputs[y]
                if (len(layer.neuron_bias) == layer.neuron_amount):
                    activation += layer.neuron_bias[y]
                    cia_archiver.log_activation(activation)
                cia_archiver.log_neuronoutput(y, x, inputs[y], layer.neuron_weights[y][x], layer.neuron_bias[y], activation)
                y=y+1
            layer_activation.append(activation)
            x=x+1
        cia_archiver.log_layeroutput(x, layer_activation)
        return layer_activation

    def output(inputs, layer):
        layer_activation = []
        
        numpy_inputs = np.array(inputs)
        numpy_inputs = numpy_inputs+layer.neuron_bias
        
        layer_activation = numpy_inputs.tolist()
        
        return layer_activation

class layer():
    class dense():
        def __init__(self, na):
            self.identifier = 'COGNITARE LAYER'
            self.type = 'DENSE'
            self.neuron_amount = na
            
            self.neuron_bias = []
            self.neuron_weights = []
                
                

class model():
    class metadata():
        layers = []
        learn_rate = 0.5
        network_loss = 0
        
        output_loss = []
        layer_outputs = []
        layer_outputs_avg = []
        activated_neurons = []
        current_output = []
    
def fit(model, input_data, expected_output, activation_function, epochs, lr):
    global experiment_weightxcost_log
    global cia_archiver
    
    logging.bold_barred('FITTING INITIALIZED ON')
    if not (check.layers()):
        return 1
    
    if not model.metadata.layers[0].neuron_bias:
        logging.new_section()
        logging.bold_barred('GENERATING NETWORK')
        net_manage.generate_bias(model)
        net_manage.generate_weight(model)
    
    def forward_propogate():
        logging.below_barred('FORWARD PROPOGATING')
        
        def clean_model():
            model.metadata.layer_outputs = []
            model.metadata.layer_outputs_avg = []
            model.metadata.activated_neurons = []
            model.metadata.current_output = []
        
        def calc_output(layer, inputs, activation_function, output_flag):  
            if not (output_flag):
                if (activation_function == 'standard'):
                    return activation.standard(inputs, layer)
            else:
                return activation.output(inputs, layer)
        
        clean_model()
        
        new_input = copy.copy(input_data)
        
        
        x=0
        while (x < len(model.metadata.layers)):
            cia_archiver.log_newlayer(x)
            if (x == len(model.metadata.layers)-1):
                new_input = calc_output(model.metadata.layers[x], new_input, activation_function, True)
                
                plt.title('New Input '+ str(x))
                plt.plot(new_input)
                plt.show()
            else:
                new_input = calc_output(model.metadata.layers[x], new_input, activation_function, False)
                
                plt.title('New Input '+ str(x))
                plt.plot(new_input)
                plt.show()
        
            model.metadata.layer_outputs.append(new_input)
             
            numpy_new_input = np.array(new_input)
            model.metadata.layer_outputs_avg.append(numpy_new_input.mean()+(numpy_new_input.mean()/2))
            
            x=x+1            
            
        model.metadata.current_output = new_input
        logging.below_barred('New Input: '+str(new_input))
        
        model.metadata.network_loss = loss.regression.RMSE(new_input, expected_output)[0] 
        model.metadata.output_loss = loss.regression.RMSE(new_input, expected_output)[1] 
        
        logging.below_barred('LOSS OF SYSTEM: '+ str(model.metadata.network_loss))
        logging.below_barred('LOSS OF OUTPUT NEURONS: ' + str(model.metadata.output_loss))
        logging.below_barred(model.metadata.layer_outputs_avg)
        
    def backward_propogate():
        global experiment_weightxcost_log
        action_array = []
        
        def new_weight(weight):
            global experiment_weightxcost_log
            
            """
            Prereq:
                1. Find out how to calculate error for hidden layers COMPLETELY
                2. Find out how to calculate the gradient
                
            
            1. First calculate error of weight through Error = weight * error * ds(x)/dx
            2. Correct weight through WeightNew = WeightPrevious + Error * input * ds(x)/dx
            """
            
            logging.below_barred('EXP - ADJUSTING WEIGHT: '+ str(weight+lr))
            
            return weight+(lr)
        
        x=0
        while (x < len(expected_output)):
            
            if (expected_output[x] > model.metadata.current_output[x]):
                action_array.append('OUTPUT NEURON - '+str(x)+' - REQUIRES INCREASE')
            else:
                if (expected_output[x] < model.metadata.current_output[x]):
                    action_array.append('OUTPUT NEURON - '+str(x)+' - REQUIRES DECREASE')
                else:
                    action_array.append('OUTPUT NEURON - '+str(x)+' - NO CHANGE')
            
            x=x+1
        
        exp_layer_select = 1

        edit_array = np.array(model.metadata.layers[exp_layer_select].neuron_weights[0])
        edit_array = edit_array + 1
        model.metadata.layers[exp_layer_select].neuron_weights[0] = edit_array.tolist()

        edit_array = np.array(model.metadata.layers[exp_layer_select+1].neuron_weights[0])
        edit_array = edit_array + 1
        model.metadata.layers[exp_layer_select+1].neuron_weights[0] = edit_array.tolist()

        logging.below_barred('WEIGHTS ADJUSTED: '+ str(model.metadata.layers[exp_layer_select].neuron_weights[0]))

        forward_propogate()
        experiment_weightxcost_log.append(model.metadata.output_loss)
        
        
        return 0
    
    forward_propogate()

    epoch = 0
    while (epoch < epochs):
        
        backward_propogate()
        epoch=epoch+1

    print('EXPERIMENT: '+ str(experiment_weightxcost_log)) 

    