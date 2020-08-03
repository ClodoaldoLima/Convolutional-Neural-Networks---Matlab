function [Features, dfativ] = cnnConv(images,W,b,strider,fativ)
    filterDim = size(W,1);
    numFilters1 = size(W,3)
    numFilters2 = size(W,4)
    numImages = size(images,4)
    imageDimX = size(images,1);
    imageDimY = size(images,2);
    convDimX = floor((imageDimX-filterDim)/strider)+1;
    convDimY = floor((imageDimY-filterDim)/strider)+1;

    Features = zeros(convDimX,convDimY,numFilters2,numImages); % Armazena as features
    dfativ=zeros(convDimX,convDimY,numFilters2,numImages); %Armazena derivada ativação
    result_conv=zeros(convDimX,convDimY,numFilters1+1); %Armazena o resultado da convolução, posição
    for i = 1:numImages
        for fil2 = 1:numFilters2
            convolvedImage = zeros(convDimX, convDimY);
            for fil1 = 1:numFilters1
                filter = squeeze(W(:,:,fil1,fil2));
                im = squeeze(images(:,:,fil1,i));
                result_conv(:,:,fil1)=conv_mod(im,filter,strider); %Realiza convolução - não rotaciona a o filtro
            end
            
            if strcmp(fativ,'relu')
                %Aplicar bias antes do maximo ou depois não altera o indice
                % max(x1+b,x2+b) = x1+b
                result_conv(:,:,1:end-1)= bsxfun(@plus,result_conv(:,:,1:end-1),b(fil2)); %soma o bias
                [convolvedImage,dconvolvedImage]=max(result_conv,[],3); %Calcula o máximo
                dfativ(:, :, fil2, i)=dconvolvedImage;                %remap 1 para 0, guarda o indice do valor do maximo 
            elseif strcmp(fativ,'sig')
                convolvedImage=sum(result_conv,3); %Soma o resultado convolução
                convolvedImage = bsxfun(@plus,convolvedImage,b(fil2)); %soma o bias
                convolvedImage = 1 ./ (1+exp(-convolvedImage));        %função ativação    
                dfativ(:, :, fil2, i)=(1-convolvedImage).*convolvedImage;
            end
            Features(:, :, fil2, i) = convolvedImage;       
        end
    end
    
end