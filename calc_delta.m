function delta_full=calc_delta(delta,W,dfativ,fativ)

[h,ne]=size(W);
[m,N]=size(dfativ);

if strcmp(fativ,'sig') %função ativação sigmoid
    delta_full = W'*(delta.*dfativ);

elseif strcmp(fativ,'relu')  %funçao ativação relu
    delta_full=zeros(ne,N);
    for i=1:ne,
        dfativ_aux=zeros(size(dfativ));
        [px,py]=find(dfativ==i);
        dfativ_aux(px,py)=1;
        for j=1:h
            delta_full(j,:)=delta_full(j,:)+ W(j,i)*(delta.*dfativ_aux(j,:));
        end
        
    end
end
