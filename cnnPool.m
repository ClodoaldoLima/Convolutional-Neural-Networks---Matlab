function [pooledFeatures,dpooledFeatures] = cnnPool1(poolDim, convolvedFeatures,strider,criterio)

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDimx = size(convolvedFeatures, 1);
convolvedDimy = size(convolvedFeatures, 2);


dimx=floor((convolvedDimx - poolDim)/strider)+1;
dimy=floor((convolvedDimy - poolDim)/strider)+1;

pooledFeatures = zeros(dimx, ...
    dimy, numFilters, numImages);

dpooledFeatures = zeros(dimx*dimy, ...
    2, numFilters, numImages);

for imageNum = 1:numImages
    for featureNum = 1:numFilters
        featuremap = squeeze(convolvedFeatures(:,:,featureNum,imageNum));
        if strcmp(criterio,'mean')
            [pooledFeaturemap,dpooledFeaturemap] = conv_mod(featuremap,ones(poolDim)/(poolDim^2),strider);
        elseif strcmp(criterio,'max')
            [pooledFeaturemap,dpooledFeaturemap]=conv_max(featuremap,ones(poolDim),strider);
        end
        pooledFeatures(:,:,featureNum,imageNum) = pooledFeaturemap;
        dpooledFeatures(:,:,featureNum,imageNum) = dpooledFeaturemap;

    end
end
end


