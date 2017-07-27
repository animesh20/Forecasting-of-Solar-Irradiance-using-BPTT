load GHI1.mat;
mae2=0;
RMSE2=0;
for i=8761:11680,
    mae2=mae2+abs(GHI1(i)-GHI1(i-5));
    RMSE2=RMSE2+((GHI1(i)-GHI1(i-5)).^2);
end;
mae2=mae2/(11680-8761);
RMSE2=sqrt(RMSE2/(11680-8761));
R=max(GHI1)-min(GHI1);
mre=mae2/(max(GHI1)-min(GHI1));