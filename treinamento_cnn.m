function cnn=treinamento_cnn(cnn,imagens,labels,testimages,testlabels)
it = 0; %numero de iterações
C = [];

% Carrega alguns parametros
epocasMax = cnn.numepocas;        %Numero de epocas
minibatch = cnn.minibatch;        %tamanho do minibatch
momIncrease = cnn.momIncrease;    %Incremento do momento
mom = cnn.mom;                 %
momentum = cnn.momentum; %termo de momento
alpha = cnn.alpha;       %taxa de aprendizado
lambda = cnn.lambda;     %coeficente de regularização
ratio = cnn.ratio;       %Taxa de congelamento dos neuronios
numCamadas = numel(cnn.camadas); %Numero de camadas


N = length(labels);              %Numero de entradas
cont=0;
for nep = 1:epocasMax
    %*****************************************************************
    % 1) Para cada epoca, realiza um congelamento de alguns neuronios
    %****************************************************************
    for l=1:numCamadas,
        camada = cnn.camadas{l};
        Idx=[];
        if (strcmp(camada.tipo,'c')) % Camada de convolução
            numFiltros = camada.numFiltros;
            camada.indcongFiltros=find(rand(numFiltros,1)<=ratio); %Indice dos filtros congelados
            Idx = camada.indcongFiltros;
            camada.txcongFiltros = length(Idx)/numFiltros; %Taxa de filtros congelados
            sprintf('Na camada Convolucao %i, foram congelados efetivamente %d filtros',l,length(Idx));
            cnn.camadas{l}=camada;
        elseif (strcmp(camada.tipo,'p')) % Camada pooling segue o mesmo congelamento da camada convolução
            sprintf('Na camada %i, foram congelados efetivamente %d filtros',l,length(Idx));
        elseif (strcmp(camada.tipo,'f')) % Congela neuronios da camada totalmente conectada
            numFiltros = camada.numhidden;
            camada.indcongFiltros=find(rand(numFiltros,1)<=ratio); %Indice dos filtros congelados
            Idx = camada.indcongFiltros;
            camada.txcongFiltros = length(Idx)/numFiltros; %Taxa de filtros congelados
            sprintf('Na camada Totalmente Conectada %i, foram congelados efetivamente %d filtros',l,length(Idx));
            cnn.camadas{l}=camada;
        end
    end
    %******************************************************************
    % Gera randomicamente os indices de entrada
    p = randperm(N);
    
    %********************************************************************
    % Separa os dados em minibatch,
    %********************************************************************
    for s=1:minibatch:(N-minibatch+1)
        it = it + 1; % Conta o numero de atualização
        %incrementa o momento
        if it == momIncrease
            mom = momentum;
        end;
        % Gera o subconjunto de treinamen
        X = imagens(:,:,:,p(s:s+minibatch-1)); %Seleciona as entrada para treinamento
        Yd = labels(p(s:s+minibatch-1));       % seleciona os rotulos para treinamento
        numImagens = size(X,4);                % [dimX, dimY, canal, numero de imagens]
        
        %*************************************************************
        %Realiza feedforward
        %************************************************************
        ativacao = X; %Ativação para camada entrada
        cont=0; %sinaliza que os dados são imagens
        for l = 1:numCamadas
            camada = cnn.camadas{l};
            if (strcmp(camada.tipo,'c')) %Camada convolução
                strider = camada.strider;
                fativ = camada.fativ;
                [ativacao, dfativ] = cnnConv(ativacao,camada.W,camada.b,strider,fativ);
                indcong=camada.indcongFiltros; %individuos congelados
                txcong=camada.txcongFiltros; %taxa de individuos
                [ativacao,dfativ]= dropout(ativacao,dfativ,indcong,txcong); %ativação para proxima camada
            elseif (strcmp(camada.tipo,'p'))  % Camada de Pooling
                strider = camada.strider;
                criterio=camada.criterio;
                [ativacao,dfativ] = cnnPool(camada.dimPool,ativacao,strider,criterio);
                if l>1
                    camada.numFiltros = cnn.camadas{l-1}.numFiltros; %adicionei este campo para facilita
                else
                    camada.numFiltros = cnn.imageCanal; %adicionei este campo para facilita
                end
            elseif (strcmp(camada.tipo,'f'))  %Camada totalmente conectada
                fativ = camada.fativ;
                save all
                [ativacao,dfativ] = cnnfull(ativacao,camada.W,camada.b,fativ);
                cont=1; %Sinaliza que os dados foram concatenados
            end
            camada.ativacao = ativacao;
            camada.dfativ=dfativ;
            cnn.camadas{l} = camada;
        end
        
        if cont==0
            %Concatena a saida anterior para as proximas camadas
            ativacao = reshape(ativacao,[],numImagens);
        end
        %camada saida
        probs = exp(bsxfun(@plus, cnn.Wd * ativacao, cnn.bd)); %exp(W*ativ + b)
        sumProbs = sum(probs, 1); % calcula a soma das exponenciais
        probs = bsxfun(@times, probs, 1 ./ sumProbs); % exp(W*ativ + b)/soma exponencial
        
        %% --------- Calcuo do Custo ----------
        % Calcula a entropia cruzada
        logp = log(probs); %Numero de classes x Numero de amostras
        index = sub2ind(size(logp),Yd',1:size(probs,2)); %busca o indice que correponde a classe
        Custo = -sum(logp(index)); %Equivalente -Yd.*log(Probs)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calcula a soma dos pesos ao quadrado
        wCusto = 0;
        for l = 1:numCamadas
            camada = cnn.camadas{l};
            if ~(strcmp(camada.tipo,'p'))
                wCusto = wCusto + sum(camada.W(:) .^ 2);
            end
        end
        % Adiciona a soma dos pesos ao quadrado
        wCusto = lambda/2*(wCusto + sum(cnn.Wd(:) .^ 2));
        CustoTotal = Custo + wCusto;   %dciona os pesos na função custo
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %---Realiza o Algoritmo Backpropagation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %softmax layer
        output = zeros(size(probs));
        output(index) = 1; %Aciona o valor de 1 a saida desejada
        DeltaSoftmax = (probs - output); %Calcula o erro
        
        
        %Transforma o erro em matriz
        %if numCamadas>1
        %    if ~strcmp(cnn.camadas{numCamadas},'f')
        %        numFiltros2 = cnn.camadas{numCamadas}.numFiltros; %Numero de filtros da ultima camada
        %    end
        %else
        %    numFiltros2 = size(X,2); %Numero de filtros da ultima camada
        %end
        
        dimSaidaX= size(cnn.camadas{numCamadas}.ativacao,1); % Numero de saida
        dimSaidaY= size(cnn.camadas{numCamadas}.ativacao,2); % Numero de saida
        camada=cnn.camadas{numCamadas}; %ultima camada        
        if strcmp(camada.tipo,'f')
            % Se tiver camada intermediaria, trabalha com matriz
            delta_ant = cnn.Wd' * DeltaSoftmax;
        else
            %Não tem camada intermediaria, transforma os dados em imagem
            %para retropropagar
            numFiltros2 = cnn.camadas{numCamadas}.numFiltros;
            delta_ant = reshape(cnn.Wd' * DeltaSoftmax,dimSaidaX,dimSaidaY,numFiltros2,numImagens);
        end
        
        % Carrega dados da camada de saida salva delta na ultima camada
        % Carrega dados da camada full salva delta na camada anterior
        % Outras camadas
        for l = numCamadas:-1:1
            camada = cnn.camadas{l}; %
            if strcmp(camada.tipo,'f')
                ativacao = cnn.camadas{l}.ativacao;
                delta_full = delta_ant;
                fativ = cnn.camadas{l}.fativ;
                dfativ = cnn.camadas{l}.dfativ;
                delta=calc_delta(delta_full,camada.W,dfativ,fativ);
                cnn.camadas{l}.delta=delta_ant.*dfativ; %passa pela função de ativação
                if l>1
                    if strcmp(cnn.camadas{l-1}.tipo,'c') | strcmp(cnn.camadas{l-1}.tipo,'p')
                        %camada anterior é convolucional
                        %Precisa transformar forma matricial para imagem
                        dimSaidaX= size(cnn.camadas{l-1}.ativacao,1); % Numero de saida
                        dimSaidaY= size(cnn.camadas{l-1}.ativacao,2); % Numero de saida
                        numFiltros = cnn.camadas{l-1}.numFiltros; %Numero de filtros
                        delta_ant = reshape(delta,dimSaidaX,dimSaidaY,numFiltros,numImagens); %camada anterior
                        
                    else
                        delta_ant = delta;
                    end
                else %camada anterior é entrada
                    dimSaidaX= size(X,1); % Numero de saida
                    dimSaidaY= size(X,2); % Numero de saida
                    numFiltros = cnn.imageCanal; %Numero de filtros
                    delta_ant = reshape(delta,dimSaidaX,dimSaidaY,numFiltros,numImagens);
                end
            elseif strcmp(camada.tipo,'p') % camada pooling
                if l>1
                    numFiltros = cnn.camadas{l-1}.numFiltros; %Numero de filtros da camada anterior
                    dimSaida1 = size(cnn.camadas{l}.ativacao,1); %dimensao da saida
                    dimSaida2 = size(cnn.camadas{l}.ativacao,2); %dimensao da saida                    
                else
                    numFiltros=cnn.imageCanal;
                    dimSaida1 = size(X,1); %dimensao da saida
                    dimSaida2 = size(X,2); %dimensao da saida                    
                end
                strider = cnn.camadas{l}.strider;
                dimPool = cnn.camadas{l}.dimPool; %dimensão do poolin
                convDim1 = dimSaida1*strider + dimPool-1;
                convDim2 = dimSaida2*strider + dimPool-1;
                criterio = cnn.camadas{l}.criterio;
                deltaPool = delta_ant;
                dfativ = cnn.camadas{l}.dfativ;
                %Unpool da ultima camada
                delta = zeros(convDim1,convDim2,numFiltros,numImagens);  %Cria
                for imNum = 1:numImagens
                    for FilterNum = 1:numFiltros
                        unpool = deltaPool(:,:,FilterNum,imNum);
                        dfativaux = dfativ(:,:,FilterNum,imNum);
                        delta(:,:,FilterNum,imNum)= delta_pool(unpool,dfativaux,dimPool,strider,convDim1,convDim2,criterio);
                    end
                end
                cnn.camadas{l}.delta = delta;
                delta_ant=delta;
            elseif strcmp(camada.tipo,'c')
                if l>1
                    numFiltros1 = cnn.camadas{l-1}.numFiltros; %Numero de filtros da camada anterior
                    dimSaida1 = size(cnn.camadas{l-1}.ativacao,1); %Numero de saida da camada atual
                    dimSaida2 = size(cnn.camadas{l-1}.ativacao,2); %Numero de saida da camada atual
                else
                    numFiltros1=cnn.imageCanal;
                    dimSaida1 = size(X,1); %Numero de saida da camada anterior
                    dimSaida2 = size(X,2); %Numero de saida da camada anterior
                end
                numFiltros2 = cnn.camadas{l}.numFiltros; %Numero de filtrso da camada posterior
                delta = zeros(dimSaida1,dimSaida2,numFiltros1,numImagens); % Matriz de derivada com zero
                deltaConv = delta_ant; %copia a derivada da camada da frente
                Wc = cnn.camadas{l}.W; %pesos da camada posterior
                strider = cnn.camadas{l}.strider;
                fativ=cnn.camadas{l}.fativ;
                dfativ=cnn.camadas{l}.dfativ;
                delta_aux=deltaConv;
                for i = 1:numImagens
                    for f1 = 1:numFiltros1
                        for f2 = 1:numFiltros2
                            %Precisa fazer convolução full com kernel
                            %rotacionado
                            df=calc_dfativ(fativ,dfativ(:,:,f2,i),f1);
                            delta_aux(:,:,f2,i) = deltaConv(:,:,f2,i).*df; %Multiplica pela derivada da função ativação
                            delta(:,:,f1,i) = delta(:,:,f1,i) +...
                                conv_full(delta_aux(:,:,f2,i),Wc(:,:,f1,f2),dimSaida1,dimSaida2,strider);
                            
                        end
                    end
                end
                cnn.camadas{l}.delta = delta_aux; %armazena na camada
                delta_ant=delta;
            end
        end
        
        %gradients
        ativacao = cnn.camadas{numCamadas}.ativacao; %ativacao da ultima camada
        %ativacao = reshape(ativacao,[],numImagens); % Transforma em vetor
        if strcmp(cnn.camadas{numCamadas}.tipo,'c') | strcmp(cnn.camadas{numCamadas}.tipo,'p')
            ativacao = reshape(ativacao,[],numImagens); % Transforma em vetor
        end            
        Wd_grad = DeltaSoftmax*(ativacao)'; %dJdw
        bd_grad = sum(DeltaSoftmax,2);%dIdb
        
        cnn.Wd_velocidade = mom*cnn.Wd_velocidade + alpha * (Wd_grad/minibatch+lambda*cnn.Wd);
        cnn.bd_velocidade = mom*cnn.bd_velocidade + alpha * (bd_grad/minibatch);
        cnn.Wd = cnn.Wd - cnn.Wd_velocidade; %atualiza os pesos da camada de saida
        cnn.bd = cnn.bd - cnn.bd_velocidade; %atualiza o bias
        
        %Realiza a atualização
        for l = numCamadas:-1:1
            camada = cnn.camadas{l};
            if(strcmp(camada.tipo,'f'))
                numhidden=camada.numhidden;
                if(l == 1)
                    ativacao = X;
                    ativacao=reshape(ativacao,[],numImagens);
                else
                    ativacao = cnn.camadas{l-1}.ativacao;
                    if ~strcmp(cnn.camadas{l-1}.tipo,'f')
                        ativacao=reshape(ativacao,[],numImagens);
                    end
                end
                Wc_grad = zeros(size(camada.W));
                bc_grad = zeros(size(camada.b));
                delta = camada.delta;
                for fil2 = 1:numhidden
                    temp = delta(fil2,:);
                    bc_grad(fil2) = sum(delta(:));
                end
                
                Wc_grad=delta*ativacao';
                camada.W_velocidade = mom*camada.W_velocidade + alpha*(Wc_grad/numImagens+lambda*camada.W);
                camada.b_velocidade = mom*camada.b_velocidade + alpha*(bc_grad/numImagens);
                camada.W = camada.W - camada.W_velocidade;
                camada.b = camada.b - camada.b_velocidade;
                
            elseif(strcmp(camada.tipo,'c'))% Camada de convolução
                numFiltros2 = camada.numFiltros;
                if(l == 1)
                    numFiltros1 = cnn.imageCanal;
                    ativacao = X;
                else
                    numFiltros1 = cnn.camadas{l-1}.numFiltros;
                    ativacao = cnn.camadas{l-1}.ativacao;
                end
                Wc_grad = zeros(size(camada.W));
                bc_grad = zeros(size(camada.b));
                DeltaConv = camada.delta;
                strider=camada.strider;
                
                for fil2 = 1:numFiltros2
                    for fil1 = 1:numFiltros1
                        for im = 1:numImagens
                            Wc_grad(:,:,fil1,fil2) = Wc_grad(:,:,fil1,fil2) +...
                                conv_comp(ativacao(:,:,fil1,im),DeltaConv(:,:,fil2,im),strider);
                        end
                    end
                    temp = DeltaConv(:,:,fil2,:);
                    bc_grad(fil2) = sum(temp(:));
                end
                camada.W_velocidade = mom*camada.W_velocidade + alpha*(Wc_grad/numImagens+lambda*camada.W);
                camada.b_velocidade = mom*camada.b_velocidade + alpha*(bc_grad/numImagens);
                camada.W = camada.W - camada.W_velocidade;
                camada.b = camada.b - camada.b_velocidade;
            end
            cnn.camadas{l} = camada;
        end
        fprintf('Epoca %d: Custo Total e Custo na iteracao %d is %f %f\n',nep,it,CustoTotal,Custo);
        C(length(C)+1) = CustoTotal;
        %break;
    end
    %cnnTest(cnn,testimages,testlabels);
   % alpha = alpha/2.0;
end
plot(C);

end
