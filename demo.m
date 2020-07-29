clear all
close all
clc
opts.alpha = 1e-1;    %taxa  de aprendizado
opts.batchsize = 50; %tamanho do conjunto de treinamento 150
opts.numepocas = 200;   %Numero de epocas
opts.imageDimX = 28;  %Dimensão do eixo X da imagem
opts.imageDimY = 28;  %Dimensão do eixo X da imagem
opts.imageCanal = 1;  %Quantidade de canais da imagem de entrada
opts.numClasses = 10; %Numero de classes
opts.lambda = 0.0001; %fator de decaimento dos pesos
opts.ratio=0.0;       %fator de congelamento dos pesos 
opts.momentum = .95;  %fator do momento
opts.mom = 0.5;       %Altera momemtno
opts.momIncrease = 20; %Numero de epocas para incrementar momento

% Carrega base de dados MINST Treinamento 
addpath .\imagens\;
images = loadMNISTImages('.\imagens\train-images-idx3-ubyte');
images = reshape(images,opts.imageDimX,opts.imageDimY,1,[]);
labels = loadMNISTLabels('.\imagens\train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10
images=images(:,:,:,1:150);
labels=labels(1:150);


% Carrega base de dados MINST Teste
testImages = loadMNISTImages('.\imagens\t10k-images-idx3-ubyte');
testImages = reshape(testImages,opts.imageDimX,opts.imageDimY,1,[]);
testLabels = loadMNISTLabels('.\imagens\t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

% Camadas
% c ---- convolucional
    %numfiltros, strider, dimFiltros, função ativação (sig, relu,)
% p ---- pooling
    % strider, dimPool, Max ou Media
% f ---- full conect 
    % fativ (sig, relu), numhidden 
    

cnn.camadas = {
%    struct('tipo', 'c', 'numFiltros', 6,'strider',2,'dimFiltros', 2,'fativ','relu') %camada convolução
%    struct('tipo', 'p', 'strider',1,'dimPool', 2,'criterio','max')                                   %camada subamostragem
    struct('tipo', 'c', 'numFiltros',8,'strider',1, 'dimFiltros', 2,'fativ','relu') %camada convolução
    struct('tipo', 'p', 'strider',2, 'dimPool',2,'criterio','max')                                   %camada subamostragem
 %   struct('tipo', 'f','fativ','sig','numhidden',100)                           %camada totalmente conectada 
 %   struct('tipo', 'f','fativ','sig','numhidden',50)
};


%Inicializa parâmetros da CNN
cnn = inicializa_parametros(cnn,opts);

treinamento_cnn(cnn,images,labels,testImages,testLabels);