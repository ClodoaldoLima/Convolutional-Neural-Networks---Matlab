%***********************************************************************
% Realiza Convolução, mas não rotaciona os kernels
%*******************************************************************
function [Conv,dConv]=conv_max(img,mask,strider)
[m1,n1]=size(img);
[m2,n2]=size(mask);

mConv = floor((m1-m2)/strider)+1;
nConv = floor((n1-n2)/strider)+1;
Conv=zeros(mConv,nConv);
dConv=zeros(mConv*mConv,2); %guarda as posições
x=1;
y=1;
cont=1;
for i=1:mConv,
    for j=1:nConv,
        img1=img(x:x+m2-1,y:y+n2-1);
        val=max(max(img1,[],2));
        [px,py]=find(img1==val);
        Conv(i,j) = val;
        dConv(cont,1)=px(1);
        dConv(cont,2)=py(1);
        y=y+strider;
        cont=cont+1;
    end
    x=x+strider;
    y=1;
end