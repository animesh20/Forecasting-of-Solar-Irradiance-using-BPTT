function [net_o,test_o,e_test]=rnn_sim(net, i_seq, o_seq, beta, R, st,o_epochs,max1,min1)
%rnn_sim    =simulates rnn
%test_o     =predicted output
%e_test    =error from the testing sequence
%net        =BPTT trained rnn network 
%i_seq      =input sequence
%o_seq      =output sequence
%beta       =learning rate for hybrid learning
%R          =needed when window size is greater than output interval
%st         =needed when window size is greater than output interval
%o_epochs   =number of times hybrid learning is to be done 
%max1,min1  =for denormalising


%dimension checking is done
[i_units,no_i]=size(i_seq);

if i_units~=net.iu,
    error('Number of input units and the input pattern size do not match.');
end;


%determining the starting and stopping index
start   =1;
stop    =no_i;


%initial hidden weights window
S_t=net.S_t;
temp=S_t;


%output initialisation
P_t=zeros(1,no_i-start+1);
Q   =zeros(1,no_i);
Ou  =zeros(1,no_i-start+1);


%testing starts here
for i=1:o_epochs,
    S_t=temp;
    for se=start:stop,
        
        %forward propagation
        S_t(:,net.interval*net.window_size+1-(net.interval-net.oi))=tanh(net.threshold1+net.u*i_seq(se)+net.v*S_t(:,net.interval*(net.window_size-1)+1));
        %P_t(se)=1/(1+exp(-(net.threshold2+net.w*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi)))));
        P_t(se)=(net.threshold2+net.w*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi)));
        
        %hybrid learning
        if(se>net.oi),
            %back-propagation
            delta=(i_seq(se)-P_t(se-net.oi));
            net.dw=beta*delta*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi))';
            net.dthreshold2=beta*delta;
            %for weight change calculation in hidden layers
            delta_hidden=delta*net.w'.*(sech(net.threshold1+net.u*i_seq(se)+net.v*S_t(:,net.interval*(net.window_size-1)+1)).^2);
            net.du=beta*delta_hidden*i_seq(se);
            net.dthreshold1=beta*delta_hidden;
            net.dv=beta*S_t(:,net.interval*(net.window_size-1)+1)*delta_hidden';
            %back-propagation through time
            for j=net.interval:net.interval:(net.interval*(net.window_size-1)),
                delta_hidden=(delta_hidden'*net.v)'.*(sech(net.threshold1+net.u*R(st-j)+net.v*S_t(:,net.interval*(net.window_size-1)+1-j)).^2);
                net.dv=net.dv+(beta*S_t(:,net.interval*(net.window_size-1)+1-j)*delta_hidden');
                net.du=net.du+(beta*delta_hidden*R(st-j));
                net.dthreshold1=net.dthreshold1+(beta*delta_hidden);
            end;
            
            %weights updation
            net.w=net.w+net.dw;
            net.v=net.v+net.dv;
            net.u=net.u+net.du;
            net.threshold1=net.threshold1+net.dthreshold1;
            net.threshold2=net.threshold2+net.dthreshold2;    
        end;
        
        %hidden weights window updation
        for j=1:(net.interval*net.window_size-(net.interval-net.oi)),
            S_t(j)=S_t(j+1);
        end;
    end;
    for r=1:no_i,
        Q(r)=(o_seq(r)*(max1-min1))+min1;
        Ou(r)=(P_t(r)*(max1-min1))+min1;
    end;
    mae=sum(abs(Ou-Q(start:stop)))/no_i;
    RMSE1 = sqrt(mean((Ou-Q(start:stop)).^2))
end;


%returning values
net.S_t=S_t;
net_o=net;
e_test=mae;
test_o=P_t;


%plotting graphs
x=1:1:500;
figure
plot(x,Q(start:start+499),x,Ou(start:start+499))
legend('Target','Predicted value')
xlabel('Number of instances')
ylabel('RSRP data')