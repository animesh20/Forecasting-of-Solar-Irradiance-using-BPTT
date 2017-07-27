z=1;
for i=1:3:70078,
    GHI1(z)=(GHI(i)+GHI(i+1)+GHI(i+2))/3;
    z=z+1;
end;