function cnn = inicializa_parametros(cnn,opts)
    numFiltros1 = opts.imageCanal; %Numero de canal - numero entrada para primeira camada
    dimInputX = opts.imageDimX;    %Dimensão da imagem original X
    dimInputY = opts.imageDimY;    %Dimensão da imagem original Y
    
    
    for l = 1:numel(cnn.camadas)
        camada = cnn.camadas{l};   % Camada convolucional
        if strcmp(camada.tipo,'c') % Camada convolucional
           numFiltros2 = camada.numFiltros;  % Numero de filtros
           dimFiltros = camada.dimFiltros;   % Dimensão do filtro
           camada.W = 1e-1*randn(dimFiltros,dimFiltros,numFiltros1,numFiltros2); %Inicializa o w
           camada.b = zeros(numFiltros2,1);  % Inicializa o valor do bias
           camada.W_velocidade = zeros(size(camada.W));
           camada.b_velocidade = zeros(size(camada.b));
                      
           dimConvX = floor((dimInputX - camada.dimFiltros)/camada.strider+1);%Dimensão da imagem gerada apos convolução
           dimConvY = floor((dimInputY - camada.dimFiltros)/camada.strider+1);%Dimensão da imagem gerada apos convolução
           camada.delta = zeros(dimConvX,dimConvY,numFiltros2,opts.batchsize); %Guarda as derivadas para todo conjunto e filtros

           numFiltros1 = numFiltros2; %Numero de filtros da proxima camda (Numrero de entradas)
           dimInputX = dimConvX; %Dimensão da imagem para proxima camada
           dimInputY = dimConvY; %Dimensão da imagem para proxima camada
 
        elseif strcmp(camada.tipo,'p')  %Camada Pooling
           dimPooledX = floor((dimInputX - camada.dimPool)/camada.strider)+1; %Dimensão da Imagem gerada pelo Pooling
           dimPooledY = floor((dimInputY - camada.dimPool)/camada.strider)+1; %Dimensão da Imagem gerada pelo Pooling
           camada.delta = zeros(dimPooledX,dimPooledY,numFiltros1,opts.batchsize); % Guarda as derivadas
           dimInputX =  dimPooledX;  %Dimensão da Imagem para proxima camada
           dimInputY =  dimPooledY;  %Dimensão da Imagem para proxima camada           
           
        elseif strcmp(camada.tipo,'f')  %Camada full
            numFiltros2 = camada.numhidden;   % Numero de neuronios da camada
            if l>1
                if ~strcmp(cnn.camadas{l-1}.tipo,'f')
                    numAtrib = numFiltros1*dimInputX*dimInputY; %Numero de atributos
                else
                    numAtrib = numFiltros1;
                end
            else
                numAtrib = numFiltros1*dimInputX*dimInputY;
            end
            camada.W = 1e-1*randn(numFiltros2,numAtrib); % hxne
            camada.b = zeros(numFiltros2,1);  % Inicializa o valor do bias
            camada.W_velocidade = zeros(size(camada.W));
            camada.b_velocidade = zeros(size(camada.b));  
            camada.delta = zeros(numFiltros2,opts.batchsize);
            numFiltros1=numFiltros2;
        end
        cnn.camadas{l} = camada;
    end
    
    if strcmp(camada.tipo,'f') %ultima camada
        cnn.quantNeuron =  numFiltros1; %Numero de entradas camada saida
    else
       cnn.quantNeuron=numFiltros1*dimInputX*dimInputY;
    end
    cnn.cost = 0;
    cnn.probs = zeros(opts.numClasses,opts.batchsize);
    
    r  = sqrt(6) / sqrt(opts.numClasses+cnn.quantNeuron+1);
    cnn.Wd = rand(opts.numClasses, cnn.quantNeuron) * 2 * r - r; % pesos da camada de saida
    cnn.bd = zeros(opts.numClasses,1); % bias da camada de saida
    cnn.Wd_velocidade = zeros(size(cnn.Wd));
    cnn.bd_velocidade = zeros(size(cnn.bd));
    cnn.delta = zeros(size(cnn.probs)); 
    
    cnn.imageDimX = opts.imageDimX;
    cnn.imageDimY = opts.imageDimY;
    cnn.imageCanal = opts.imageCanal; %Quantidade de canais na imagem
    cnn.numClasses = opts.numClasses; %Numero de classes
    cnn.alpha = opts.alpha;           %taxa de aprendizado
    cnn.minibatch = opts.batchsize;   %tamanho do batch
    cnn.numepocas = opts.numepocas;   %numero de epocas
    cnn.lambda = opts.lambda;
    cnn.momentum = opts.momentum;
    cnn.mom = opts.mom;
    cnn.momIncrease = opts.momIncrease;
    cnn.ratio = opts.ratio;
end