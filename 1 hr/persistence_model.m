load GHI1.mat;
mae2=0;
RMSE2=0;
for i=4381:5840,
    mae2=mae2+abs(GHI1(i)-GHI1(i-2));
    RMSE2=RMSE2+((GHI1(i)-GHI1(i-2)).^2);
end;
mae2=mae2/(5840-4381);
mre=mae2/(max(GHI1)-min(GHI1));
RMSE2=sqrt(RMSE2/(5840-4381));
R=max(GHI1)-min(GHI1);

