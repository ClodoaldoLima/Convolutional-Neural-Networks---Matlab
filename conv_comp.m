function Conv=conv_comp(img,mask,strider)
[m1,n1]=size(img);
[m2,n2]=size(mask);
if strider>1
    tamx = floor(m1/m2); %tamanho do filtro x
    tamy = floor(n1/n2); %tamanho do filtro y
    dimx=strider*m2+rem((m1-tamx),strider)-1;
    dimy=strider*n2+rem((n1-tamy),strider)-1;
    maskaux=mask;
    mask=zeros(dimx,dimy);
    ix=1;
    for i=1:strider:strider*m2
        jx=1;
        for j=1:strider:strider*n2
            mask(i,j)=maskaux(ix,jx);
            jx=jx+1;
        end
        ix=ix+1;
    end
    m2 = dimx;
    n2 = dimy;
end

mConv = (m1-m2+1);
nConv = (n1-n2+1);
Conv=zeros(mConv,nConv);
x=1;
y=1;
for i=1:mConv,
    for j=1:nConv,
        img1=img(x:x+m2-1,y:y+n2-1);
        Conv(i,j)=sum(sum(img1.*mask));
        y=y+1;
    end
    x=x+1;
    y=1;
end