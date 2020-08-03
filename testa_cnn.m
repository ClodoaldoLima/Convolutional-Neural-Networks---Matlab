function testa_cnn(cnn,imagens,labels)

numImagens = length(imagens);%Numero de Imagens
ativacao = imagens;           %Imagens como entrada
numCamadas = size(cnn.camadas);%Numero de camadas

fprintf('Numero total de Imagens igual a %d',numImagens)
cont=0; %sinaliza que os dados são imagens
for l = 1:numCamadas
    camada = cnn.camadas{l};
    if (strcmp(camada.tipo,'c')) %Camada convolução
        strider = camada.strider;
        fativ = camada.fativ;
        [ativacao, dfativ] = cnnConv(ativacao,camada.W,camada.b,strider,fativ);
    elseif (strcmp(camada.tipo,'p'))  % Camada de Pooling
        strider = camada.strider;
        criterio=camada.criterio;
        [ativacao,dfativ] = cnnPool(camada.dimPool,ativacao,strider,criterio);
    elseif (strcmp(camada.tipo,'f'))  %Camada totalmente conectada
        fativ = camada.fativ;
        [ativacao,dfativ] = cnnfull(ativacao,camada.W,camada.b,fativ);
        cont=1; %Sinaliza que os dados foram concatenados
    end
end

if cont==0
    %Concatena a saida anterior para as proximas camadas
    ativacao = reshape(ativacao,[],numImagens);
end

%camada saida
probs = exp(bsxfun(@plus, cnn.Wd * ativacao, cnn.bd)); %exp(W*ativ + b)
sumProbs = sum(probs, 1); % calcula a soma das exponenciais
probs = bsxfun(@times, probs, 1 ./ sumProbs); % exp(W*ativ + b)/soma exponencial

[~,preds] = max(probs,[],1);
preds = preds';

acc = sum(preds==labels)/length(preds);
fprintf('\nAcurácia igual a %f\n',acc);
label=unique([unique(labels);unique(preds)]); %rotulos atribuidos
numlabel = length(label);
MatrizConf = zeros(numlabel,numlabel);
for i=1:numlabel,
    for j=1:numlabel,
        px=find(labels==i);
        if ~isempty(px)
            MatrizConf(i,j)=length(find(preds(px)==j));
        end
    end
end
   
save('arq','MatrizConf','cnn');
save all
end