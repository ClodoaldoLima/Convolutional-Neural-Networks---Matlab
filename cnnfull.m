function [ativacao,dativacao]=cnnfull(ativacao,W,b,fativ)
[m,N,numFiltro,numImagens]= size(ativacao);

if numImagens>1 %As imagens ainda já foram concatenadas
    %Concatena a saida anterior para as proximas camadas
    ativacao = reshape(ativacao,[],numImagens);
    N=numImagens;
end
if strcmp(fativ,'sig')
    Zin=bsxfun(@plus, W * ativacao, b); %h x N
    ativacao=1./(1+exp(-Zin)); %n x N
    dativacao=(1-ativacao).*ativacao;
elseif strcmp(fativ,'relu')
    [h,ne]=size(W);
    for i=1:h
        for j=1:ne
            Zin(j,:) = W(i,j)*ativacao(j,:);
        end
        [val,pos]=max([zeros(1,N);Zin],[],1);
        Z(i,:)=val;
        dZ(i,:)=pos-1; %remap 1 para 0
    end
    ativacao=Z;
    dativacao=dZ;
end
end