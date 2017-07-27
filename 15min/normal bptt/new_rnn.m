function net=new_rnn(iu,hu,ou,eta,window_size,interval,output_interval)
%iu=number of input units
%hu=number of hidden units
%ou=number of output units
%eta=learning rate
%window_size=number of times the network is to be unfolded
%interval=the interval at which inputs are taken for the unfolded network
%output_interval=the interval at which output is to be predicted


%setting up the parameters for the network
net.iu              =iu;
net.hu              =hu;
net.ou              =ou;
net.eta             =eta;
net.window_size     =window_size;
net.interval        =interval;
net.oi              =output_interval;


%initialising the biases 
%threshold1=bias unit to hidden layer
%threshold2=bias unit to output layer
net.threshold1=zeros(hu,1);
net.threshold2=zeros(ou,1);


%initialising weights
net.u=(1/sqrt(iu))*(2*rand(hu,iu)-1);   %from input layer to hidden layer
net.v=(1/sqrt(hu))*(2*rand(hu,hu)-1);   %from hidden layer to hidden layer
net.w=(1/sqrt(hu))*(2*rand(ou,hu)-1);   %from hidden layer to output layer


%initialising delta terms
net.dthreshold1=zeros(hu,1);
net.dthreshold2=zeros(ou,1);
net.du=zeros(hu,iu);
net.dv=zeros(hu,hu);
net.dw=zeros(ou,hu);


%define the initial hidden weights window
net.S_t=2*rand(hu,net.interval*net.window_size+1-(net.interval-net.oi))-1;