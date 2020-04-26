function [lgraph] =prepareTransferLearningLayers(baseNetName,numClasses)
% prepareTransferLearningLayers  Prepare a pre-trained Deep Learning model
% for Transfer Learning based on any of the pre-trained networks available
% in MATLAB. Follows the approach outlined in https://uk.mathworks.com/help/nnet/examples/transfer-learning-using-googlenet.html
% but for any desired base network from the available base models (i.e., rather than just for googlenet as per the reference example). 
%
%   Inputs:
%       baseNetName  Name of the pre-trained network (see https://uk.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html) 
%                    Note: this is identical to the network name for the base network in matlab except for googlenet trained on Placed365 whereby the argument to be specified is googlenetplaces
%       numClasses   Number of target classes for Transfer Learning
%
%   Outputs:
%       lgraph       Modified layer graph ready for training with new
%                    images
%
% Created by Yusuf Jafry with MATLAB R2020a

if strcmp(baseNetName,"alexnet")
    net =alexnet;
    lgraph = layerGraph(net.Layers);%(net);
    %Replace final layers
    layersToRemove={'fc8' 'prob' 'output'};
    %Create new layers
    newLayers = [
     fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
     softmaxLayer('Name','softmax')
     classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='drop7';
    newLayerToConnect='fc';    
elseif strcmp(baseNetName,"vgg16")
    net =vgg16;
    lgraph = layerGraph(net.Layers);%(net);
    %Replace final layers
    layersToRemove={'fc8' 'prob' 'output'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='drop7';
    newLayerToConnect='fc';       
elseif strcmp(baseNetName,"vgg19")
    net =vgg19;
    lgraph = layerGraph(net.Layers);%(net);
    %Replace final layers
    layersToRemove={'fc8' 'prob' 'output'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='drop7';
    newLayerToConnect='fc';    
elseif strcmp(baseNetName,"darknet53")
    net =darknet53;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove={'conv53' 'softmax' 'output'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='avg1';
    newLayerToConnect='fc'; 
 elseif strcmp(baseNetName,"darknet19")
    net =darknet19;
    lgraph = layerGraph(net.Layers);%(net);
    %Replace final layers
    layersToRemove={'conv19' 'avg1' 'softmax' 'output'};
    %Create new layers
    newLayers = [
    convolution2dLayer(1,numClasses,'Name','conv19','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='leaky18';
    newLayerToConnect='conv19';
 elseif strcmp(baseNetName,"densenet201")
    net =densenet201;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove={'fc1000' 'fc1000_softmax' 'ClassificationLayer_fc1000'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='avg_pool';
    newLayerToConnect='fc';              
 elseif strcmp(baseNetName,"googlenet")
    net =googlenet;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove={'loss3-classifier' 'prob' 'output'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='pool5-drop_7x7_s1';
    newLayerToConnect='fc';
 elseif strcmp(baseNetName,"googlenetplaces")
    net = googlenet('Weights','places365');
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove={'loss3-classifier' 'prob' 'output'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='pool5-drop_7x7_s1';
    newLayerToConnect='fc';
elseif strcmp(baseNetName,"inceptionresnetv2")
    net = inceptionresnetv2;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove={'predictions' 'predictions_softmax' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='avg_pool';
    newLayerToConnect='fc';    
elseif strcmp(baseNetName,"inceptionv3")
    net = inceptionv3;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'predictions' 'predictions_softmax' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='avg_pool';
    newLayerToConnect='fc';    
elseif strcmp(baseNetName,"mobilenetv2")
    net = mobilenetv2;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'Logits' 'Logits_softmax' 'ClassificationLayer_Logits'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='global_average_pooling2d_1';
    newLayerToConnect='fc';    
elseif strcmp(baseNetName,"nasnetmobile")
    net = nasnetmobile;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'predictions' 'predictions_softmax' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='global_average_pooling2d_1';
    newLayerToConnect='fc';    
elseif strcmp(baseNetName,"nasnetlarge")
    net = nasnetlarge;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'predictions' 'predictions_softmax' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='global_average_pooling2d_2';
    newLayerToConnect='fc';        
elseif strcmp(baseNetName,"resnet18")
    net = resnet18;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'fc1000' 'prob' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='pool5';
    newLayerToConnect='fc';        
elseif strcmp(baseNetName,"resnet50")
    net = resnet50;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'fc1000' 'fc1000_softmax' 'ClassificationLayer_fc1000'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='avg_pool';
    newLayerToConnect='fc';   
elseif strcmp(baseNetName,"resnet101")
    net = resnet101;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'fc1000' 'prob' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='pool5';
    newLayerToConnect='fc';            
elseif strcmp(baseNetName,"shufflenet")
    net = shufflenet;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'node_202' 'node_203' 'ClassificationLayer_node_203'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];    
    %For connecting old to new...
    originalLayerToConnect='node_200';
    newLayerToConnect='fc';            
elseif strcmp(baseNetName,"squeezenet")
    net = squeezenet;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'conv10' 'relu_conv10' 'pool10' 'prob' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    convolution2dLayer(1,numClasses,'Name','conv10','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu_conv10')           
    globalAveragePooling2dLayer('Name','pool10')
    softmaxLayer('Name','prob')
    classificationLayer('Name','ClassificationLayer_predictions')];
    %For connecting old to new...
    originalLayerToConnect='drop9';
    newLayerToConnect='conv10';   
elseif strcmp(baseNetName,"xception")
    net = xception;
    lgraph = layerGraph(net);
    %Replace final layers
    layersToRemove= {'predictions' 'predictions_softmax' 'ClassificationLayer_predictions'};
    %Create new layers
    newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
    %For connecting old to new...
    originalLayerToConnect='avg_pool';
    newLayerToConnect='fc';          
end        
%Make the adjustments
lgraph = removeLayers(lgraph, layersToRemove);
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,originalLayerToConnect,newLayerToConnect);
end