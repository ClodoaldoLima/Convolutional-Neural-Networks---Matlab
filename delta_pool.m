function deltaConv=delta_pool(delta,dConv,dimPool,strider,dimx,dimy,criterio)
[m,n]=size(delta);
deltaConv=zeros(dimx,dimy); %tamanho do pooling

if strcmp(criterio,'max')
    for i=1:m
        for j=1:n
            p=n*(i-1)+j;
            px = dConv(p,1);
            py = dConv(p,2);
            Ix=px+strider*(i-1);
            Iy=py+strider*(j-1);
            deltaConv(Ix,Iy) = deltaConv(Ix,Iy)+ delta(i,j);
        end
    end   
    
elseif strcmp(criterio,'mean')
    for i=1:m
        for j=1:n
            for kx=1:dimPool
                for ky=1:dimPool
                    Ix=kx+strider*(i-1);
                    Iy=ky+strider*(j-1);
                    deltaConv(Ix,Iy) = deltaConv(Ix,Iy)+ delta(i,j);
                end
            end
        end
    end
    
end
end
