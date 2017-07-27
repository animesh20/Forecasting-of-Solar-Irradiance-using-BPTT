load GHI1.mat;
mae2=0;
RMSE2=0;
for i=17521:23360,
    mae2=mae2+abs(GHI1(i)-GHI1(i-4));
    RMSE2=RMSE2+((GHI1(i)-GHI1(i-4)).^2);
end;
mae2=mae2/(23360-17521);
RMSE2=sqrt(RMSE2/(23360-17521));
R=max(GHI1)-min(GHI1);
mre=mae2/(max(GHI1)-min(GHI1));