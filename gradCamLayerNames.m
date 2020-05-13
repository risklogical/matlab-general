function [softmaxName,featureLayerName] = gradCamLayerNames(netName) 
% gradCamLayerNames  Retrieve the relevant layer names (for use in gradcam) from any of 
% the pre-trained Deep Learning networks available in MATLAB. Follows the approach outlined in https://uk.mathworks.com/help/deeplearning/ug/gradcam-explains-why.html
% but for any desired network from those available (i.e., rather than just for googlenet as per the reference example). 
%
%   Inputs:
%       netName           Name of the pre-trained network (see https://uk.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html) 
%
%   Outputs:
%       softmaxName       Name of the softmax layer for use with the gradcam function
%       featureLayerName  Name of the feature layer for use with the gradcam function
%
%
% Created by Yusuf Jafry with MATLAB R2020a

if netName == "squeezenet"
  softmaxName = 'prob';
  %Specify either the last ReLU layer with non-singleton spatial dimensions, or the last layer that gathers the outputs of ReLU layers (such as a depth concatenation or an addition layer). If your network does not contain any ReLU layers, specify the name of the final convolutional layer that has non-singleton spatial dimensions in the output.
  featureLayerName = 'relu_conv10';    
elseif netName == "googlenet" ||  netName == "googlenetplaces"
  softmaxName = 'softmax';
  featureLayerName = 'inception_5b-output';
elseif netName == "resnet18"  
  softmaxName = 'softmax';
  featureLayerName = 'res5b_relu';    
elseif netName == "mobilenetv2"
  softmaxName = 'Logits_softmax';
  featureLayerName = 'out_relu';      
elseif netName == "alexnet" 
  softmaxName = 'prob';
  featureLayerName = 'relu5';     
elseif netName == "vgg16" 
  softmaxName = 'prob';
  featureLayerName = 'relu5_3';   
elseif netName == "vgg19" 
  softmaxName = 'prob';
  featureLayerName = 'relu5_4';   
elseif netName == "darknet19" 
  softmaxName = 'softmax';
  featureLayerName = 'leaky18';   
elseif netName == "xception"     
  softmaxName = 'predictions_softmax';
  featureLayerName = 'block14_sepconv2_act';    
elseif netName == "shufflenet"     
  softmaxName = 'node_203';
  featureLayerName = 'node_199';   
elseif netName == "nasnetlarge"     
  softmaxName = 'predictions_softmax';
  featureLayerName = 'activation_520';    
elseif netName == "nasnetmobile"     
  softmaxName = 'predictions_softmax';
  featureLayerName = 'activation_188';    
elseif netName == "inceptionresnetv2" 
  softmaxName = 'predictions_softmax';
  featureLayerName = 'conv_7b_ac';      
elseif netName == "inceptionv3" 
  softmaxName = 'predictions_softmax';
  featureLayerName = 'mixed10';   
elseif netName == "darknet53" 
  softmaxName = 'softmax';
  featureLayerName = 'res23';   
elseif netName == "densenet201" 
  softmaxName = 'fc1000_softmax';
  featureLayerName = 'bn';  
elseif netName == "resnet50"    
  softmaxName = 'fc1000_softmax';
  featureLayerName = 'activation_49_relu';    
elseif netName == "resnet101"    
  softmaxName = 'prob';
  featureLayerName = 'res5c_relu';    
    
end

end
