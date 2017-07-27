load GHI.mat;
z=1;
for i=1:2:23359,
    GHI1(z)=(GHI(i)+GHI(i+1))/2;
    z=z+1;
end;