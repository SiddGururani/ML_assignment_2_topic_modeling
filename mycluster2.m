function [ Pw_z, Pd_z, Pz ] = mycluster2( bow, K )
%
% Your goal of this assignment is implementing your own text clustering algo.
%
% Input:
%     bow: data set. Bag of words representation of text document as
%     described in the assignment.
%
%     K: the number of desired topics/clusters. 
%
% Output:
%     class: the assignment of each topic. The
%     assignment should be 1, 2, 3, etc. 
%
% For submission, you need to code your own implementation without using
% any existing libraries

% YOUR IMPLEMENTATION SHOULD START HERE!

[ndata, dim] = size(bow);

%% Randomly initializing the matrices

Pw_z = rand(dim,K);
Pd_z = rand(ndata,K);
Pz = rand(K,1);

for i = 1:K
        Pw_z(:,i) = Pw_z(:,i)./norm(Pw_z(:,i),1);
end
for i = 1:K
        Pd_z(:,i) = Pd_z(:,i)./norm(Pd_z(:,i),1);
end
Pz = Pz./norm(Pz,1);

Pdw = zeros(ndata,dim);
for d = 1:ndata
    for i = 1:K
        Pdw(d,:) = Pdw(d,:) + (Pz(i).*Pd_z(d,i).*Pw_z(:,i))';
    end
end

Pz_dw = cell(K,1);
for i = 1:K
    Pz_dw{i} = zeros(ndata, dim);
end

ll0 = Inf;
ll1 = 0;

%% Iterations
tic;
for i = 1:30
    i
%     if(abs(ll0 - ll1) < 1e-5)
%         break;
%     end
%     ll0 = ll1;
    %% Expectation step
    for d = 1:ndata
        w = find(bow(d,:));
        for z = 1:K
            Pz_dw{z}(d,w) = (Pz(z).*Pd_z(d,z).*Pw_z(w,z))'./Pdw(d,w); %P_z|d,w update
        end
    end
    
    %% Maximization step
    for z = 1:K
        for w = 1:dim
            d = find(bow(:,w));
            Pw_z(w,z) = sum(bow(d,w).*Pz_dw{z}(d,w)); %P_d|z update
        end
        Pz(z) = sum(Pw_z(:,z));
        Pw_z(:,z) = Pw_z(:,z)/Pz(z);
    end
    for z = 1:K
        for d = 1:ndata
            w = find(bow(d,:));
            Pd_z(d,z) = sum(bow(d,w).*Pz_dw{z}(d,w)); %P_w|z update
        end
        %assert(sum(Pd_z(:,z)) - Pz(z) < 1e-10);
        Pd_z(:,z) = Pd_z(:,z)/Pz(z);
    end
    
    Pz = Pz/sum(Pz); %P_z update
    
    %% Recompute Log-likelihood
%     ll1 = 0;
%     Pdw = zeros(ndata,dim);
%     for d = 1:ndata
%         for i = 1:K
%             Pdw(d,:) = Pdw(d,:) + (Pz(i).*Pd_z(d,i).*Pw_z(:,i))';
%         end
%         w = find(bow(d,:));
%         ll1 = ll1 + sum(bow(d,w) .* log(Pdw(d,w)));
%     end
%     ll1
end
toc
end
