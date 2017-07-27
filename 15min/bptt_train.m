function [net_o,e_validate]=bptt_train(net, i_seq, o_seq, epochs,i_seq1,o_seq1,st,beta,R,max1,min1)
%net_o      =network trained using bptt
%e_validate =error from the validation sequence
%net        =initialised rnn network 
%i_seq      =input sequence
%o_seq      =output sequence
%epochs     =number of iterations for which network is trained
%i_seq1      =input sequence
%o_seq1      =output sequence
%beta       =learning rate for hybrid learning
%R          =needed when window size is greater than output interval
%st         =needed when window size is greater than output interval
%max1,min1  =needed for denormalising


%For training
%dimension checking is done
tic
[i_units,no_i]=size(i_seq);
[o_units,no_o]=size(o_seq);

if i_units~=net.iu,
    error('Number of input units and the input pattern size do not match.');
end;
if o_units~=net.ou,
    error('Number of output units and the output pattern size do not match.');
end;
if no_o~=no_i,
    error('The input and the output pattern size do not match.');
end;


%determining the starting and stopping index of training
start   =net.interval*net.window_size;
stop    =no_i;


%initial hidden weights window
S_t=net.S_t;
temp=S_t;


%output initialisation
P_t=zeros(1,no_i-start+1);
Q   =zeros(1,no_i);
Ou  =zeros(1,no_i-start+1);


%For validation
%dimension checking is done
[i_units1,no_i1]=size(i_seq1);

if i_units1~=net.iu,
    error('Number of input units and the input pattern size do not match.');
end;


%determining the starting and stopping index
start1   =1;
stop1    =no_i1;


%output initialisation
P_t1=zeros(1,no_i1-start1+1);
Q1   =zeros(1,no_i1);
Ou1  =zeros(1,no_i1);
mae1 =0;
mae  =0;

%computation
for i=1:epochs,
    mae1=mae;
    S_t=temp;
    %training starts here
    for se=start:stop,
        a=se-start+1;
        
        %forward propagation
        S_t(:,net.interval*net.window_size+1-(net.interval-net.oi))=tanh(net.threshold1+net.u*i_seq(se)+net.v*S_t(:,net.interval*(net.window_size-1)+1));
        %P_t(a)=1/(1+exp(-(net.threshold2+net.w*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi)))));
        P_t(a)=(net.threshold2+net.w*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi)));
        
        %back-propagation
        delta=(o_seq(se)-P_t(a));
        %for hybrid learning in training phase
        %if(a>net.oi),
            %delta=delta+(i_seq(se)-P_t(a-net.oi));
        %end;
        net.dw=net.eta*delta*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi))';
        net.dthreshold2=net.eta*delta;
        %for weight change calculation in hidden layers
        delta_hidden=delta*net.w'.*(sech(net.threshold1+net.u*i_seq(se)+net.v*S_t(:,net.interval*(net.window_size-1)+1)).^2);
        net.du=net.eta*delta_hidden*i_seq(se);
        net.dthreshold1=net.eta*delta_hidden;
        net.dv=net.eta*S_t(:,net.interval*(net.window_size-1)+1)*delta_hidden';
        %back-propagation through time
        for j=net.interval:net.interval:(net.interval*(net.window_size-1)),
            delta_hidden=(delta_hidden'*net.v)'.*(sech(net.threshold1+net.u*i_seq(se-j)+net.v*S_t(:,net.interval*(net.window_size-1)+1-j)).^2);
            net.dv=net.dv+(net.eta*S_t(:,net.interval*(net.window_size-1)+1-j)*delta_hidden');
            net.du=net.du+(net.eta*delta_hidden*i_seq(se-j));
            net.dthreshold1=net.dthreshold1+(net.eta*delta_hidden);
        end;
        
        %weights updation
        net.w=net.w+net.dw;
        net.v=net.v+net.dv;
        net.u=net.u+net.du;
        net.threshold1=net.threshold1+net.dthreshold1;
        net.threshold2=net.threshold2+net.dthreshold2;
        
        %hidden weights window updation
        for j=1:(net.interval*net.window_size-(net.interval-net.oi)),
            S_t(j)=S_t(j+1);
        end;
    end;
    for r=1:no_i,
        Q(r)=(o_seq(r)*(max1-min1))+min1;
    end;
    for r=1:no_i-start+1,
        Ou(r)=(P_t(r)*(max1-min1))+min1;
    end;
    mae=sum(abs(Ou-Q(start:stop)))/no_i;
    
    %validation starts here
    for se=start1:stop1,
        
        %forward propagation
        S_t(:,net.interval*net.window_size+1-(net.interval-net.oi))=tanh(net.threshold1+net.u*i_seq1(se)+net.v*S_t(:,net.interval*(net.window_size-1)+1));
        %P_t1(se)=1/(1+exp(-(net.threshold2+net.w*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi)))));
        P_t1(se)=(net.threshold2+net.w*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi)));
        
        %hybrid learning
        if(se>net.oi),
            %back-propagation
            delta=(i_seq1(se)-P_t1(se-net.oi));
            net.dw=beta*delta*S_t(:,net.interval*net.window_size+1-(net.interval-net.oi))';
            net.dthreshold2=beta*delta;
            %for weight change calculation in hidden layers
            delta_hidden=delta*net.w'.*(sech(net.threshold1+net.u*i_seq(se)+net.v*S_t(:,net.interval*(net.window_size-1)+1)).^2);
            net.du=beta*delta_hidden*i_seq1(se);
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
    for r=1:no_i1,
        Q1(r)=(o_seq1(r)*(max1-min1))+min1;
        Ou1(r)=(P_t1(r)*(max1-min1))+min1;
    end;
    mae=sum(abs(Ou1-Q1(start1:stop1)))/no_i1;
    fprintf('\nThe validation error for iteration ');
    fprintf('%d',i);
    fprintf(' is: ');
    fprintf('%d',mae);
    fprintf('\n');
    RMSE = sqrt(mean((Ou1-Q1(start1:stop1)).^2))
    
    if(abs(mae1-mae)<1e-02),
        break;
    end;
end;


%returning values
net.S_t=S_t;
net_o=net;
e_validate=mae;


%plotting graphs
%Training
x=1:1:500;
figure
plot(x,Q(stop-499:stop),x,Ou(a-499:a))
legend('Target','Predicted value')
xlabel('Number of instances')
ylabel('GHI data')

%validation
x=1:1:500;
figure
plot(x,Q1(start1:start1+499),x,Ou1(start1:start1+499))
legend('Target','Predicted value')
xlabel('Number of instances')
ylabel('GHI data')
toc