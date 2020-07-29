function Conv=conv_full(img,mask,dimX,dimY,strider)
mask=rot90(mask,2); %Rotaciona o kernel

[m1,n1]=size(img);
[m2,n2]=size(mask);

tamx = m1+2*m2-2+(strider-1)*(m1-1); %adciona zeros antes e no meio
tamy = n1+2*n2-2+(strider-1)*(n1-1); %adciona zeros antes e no meio

img1=zeros(tamx,tamy);

kx=0;
for i=m2:strider:m2+strider*(m1-1)
    kx=kx+1;
    ky=1;
    for j=n2:strider:n2+strider*(n1-1)
        img1(i,j)=img(kx,ky);
        ky=ky+1;
    end
end

strider=1; %independente do strider de entrada, a convolução sera strider 1
[m1,n1]=size(img1);
mConv = floor((m1-m2)/strider)+1;
nConv = floor((n1-n2)/strider)+1;

Conv=zeros(dimX,dimY);
x=1;
y=1;
for i=1:mConv,
    for j=1:nConv,
        img2=img1(x:x+m2-1,y:y+n2-1);
        Conv(i,j)=sum(sum(img2.*mask));
        y=y+strider;
    end
    x=x+strider;
    y=1;
end