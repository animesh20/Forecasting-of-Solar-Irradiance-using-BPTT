
function net0=project()

%defining parameters
interval            =10;
output_i            =5;
epochs              =30;
epoch_inter         =30;
iu                  =1;
hu                  =100;
ou                  =1;
eta                 =0.0001;
beta                =0.0001;
window_size         =10;
initial_epoch_value =30;
o_epochs            =1;

%initialisation
load s1.mat;
rng(s);

%creating a new network
net=new_rnn(iu,hu,ou,eta,window_size,interval,output_i);

%loading GHI(PSP) (Watts per square meter)
load GHI1.mat;
%GHI=GHI';
max1=max(GHI1(1:4380));
min1=min(GHI1(1:4380));
%min1=0;
%max1=2000;
for i=1:length(GHI1),
    GHI1(i)=(GHI1(i)-min1)/(max1-min1);
end;
figure
plot(GHI1(1:500));

%training dataset
input= GHI1(1:2920-output_i);
output= GHI1(1+output_i:2920);

%validation dataset
input1= GHI1(2921-output_i:4380-output_i);
output1= GHI1(2921:4380);

%testing dataset
input2= GHI1(4381-output_i:5840-output_i);
output2= GHI1(4381:5840);


%performance computation
validation_error=zeros(1,epochs/epoch_inter);
test_error=zeros(1,epochs/epoch_inter);
y=1;

for i=initial_epoch_value:epoch_inter:epochs,
    
    %training
    start=2921-output_i;
    [net1,validation_error(y)]=bptt_train(net,input,output,i,input1,output1,start,beta,GHI1,max1,min1);
    
    %testing
    start= 4381-output_i;
    [net2,test_o,test_error(y)]=rnn_sim(net1,input2,output2, beta, GHI1, start,o_epochs,max1,min1);
    
    y=y+1;
end;

%displaying error
test_error

%plotting the errors
%x=initial_epoch_value:epoch_inter:epochs;
%figure
%plot(x,train_error)
%xlabel('number of epochs')
%ylabel('training error')
%figure
%plot(x,validation_error)
%xlabel('number of epochs')
%ylabel('validation error')
%figure
%plot(x,test_error)
%xlabel('number of epochs')
%ylabel('testing error')