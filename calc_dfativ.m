function df=calc_dfativ(fativ,dfativ,numFiltro)


if strcmp(fativ,'sig') %fun��o de ativa��o sigmoid
    df = dfeativ;
elseif strcmp(fativ,'relu')
    dfativ_aux=zeros(size(dfativ));
    [px,py]=find(dfativ==numFiltro);
    dfativ_aux(px,py)=1;
    df=dfativ_aux;
end
