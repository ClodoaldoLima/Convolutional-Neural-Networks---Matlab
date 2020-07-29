%***********************************************************************
% Realiza Convolução, mas não rotaciona os kernels
%*******************************************************************
function [Conv,dConv]=conv_mod(img,mask,strider)
[m1,n1]=size(img);
[m2,n2]=size(mask);

mConv = floor((m1-m2)/strider)+1;
nConv = floor((n1-n2)/strider)+1;
Conv=zeros(mConv,nConv);
x=1;
y=1;
for i=1:mConv,
    for j=1:nConv,
        img1=img(x:x+m2-1,y:y+n2-1);
        Conv(i,j)=sum(sum(img1.*mask));
        y=y+strider;
    end
    x=x+strider;
    y=1;
end
dConv = zeros(mConv,nConv); %não precisa armazenar a derivada de convolução