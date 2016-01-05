function [ class ] = mycluster( bow, K )
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

%% Initializing topic assignment using K-medoids clustering on dataset
 [class, ~] = kmedoids(bow, K);

%% Initializing topic assignment using K-means clustering on dataset
%[class, ~] = kmeans(bow, K);

%% Randomly initializing topic assignment for documents
% class = floor(K*rand(ndata,1))+1;

%% Initializing topic model parameters

pi = zeros(1,K);
mu = zeros(dim,K);
gamma = zeros(ndata, K);
for i=1:K
    gamma((class == i),i) = 1;
    mu(:,i) = sum(bow((class == i),:))/sum(sum(bow((class == i),:)));
end
pi = sum(gamma)/ndata;

%% Performing expectation maximization

logl = Inf;

for iter = 1:500
    logl1 = 0.0;
    iter = iter+1;
    % E-step: Recomputing cluster/topic assignment
    for i = 1:ndata
        gam_norm = 0.0;
        for c = 1:K
            gamtmp = pi(c)*prod(bsxfun(@power, mu(:,c), bow(i,:)'));
            gamma(i,c) = gamtmp;
            gam_norm = gam_norm + gamtmp;
        end
        gamma(i,:) = gamma(i,:)/gam_norm;
        logl1 = logl1 + gam_norm;
    end
    logl1
    % M-step: Recomputing parameters
    pi = sum(gamma)/ndata;
    for c = 1:K
        mu(:,c) = gamma(:,c)'*bow(:,:);
        mu(:,c) = mu(:,c)/sum(mu(:,c));
    end
    if(abs(logl1 - logl) == 0)
        break;
    end
    logl = logl1;
end
[~, class] = max(gamma,[],2);
