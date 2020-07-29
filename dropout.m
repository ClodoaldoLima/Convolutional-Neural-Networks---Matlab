%*************************************************************
%  y - ativação do neuronio
% dy - derivada da função de ativação do neuronio
% idx - indices dos neuronios congelados
%*************************************************************

function [y,dy]= dropout(y,dy,idx,tx)

[dimX,dimY,numFilters, numImagens] = size(y);
num = length(idx);
cont=1;
for i=1:numFilters
    for j=1:numImagens,
        if cont<=num,
            if idx(cont)== i %neuronio foi congelado
                y(:,:,i,j) = zeros(dimX,dimY); %ativação vai para zero
                dy(:,:,i,j)=zeros(dimX,dimY); %derivada vai para zero
            else
                y(:,:,i,j)=y(:,:,i,j).*(ones(dimX,dimY)/(1-tx)); % aumenta a saida
            end
        else
            y(:,:,i,j)=y(:,:,i,j).*(ones(dimX,dimY)/(1-tx)); % aumenta a saida
        end
    end
    if cont<=num,
        if idx(cont)== i
            cont = cont+1;
        end
    end
end