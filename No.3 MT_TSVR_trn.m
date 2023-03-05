function [alpha, b] = MT_TSVR_trn(trnX, trnY, gamma, lambda, p, tau)

[l, m] = size(trnY); 
matrix = find_temporal_matrix(size(trnY,1),tau); 

K = Kerfun('rbf', trnX, trnX, p, 0);
matrix2 = [];
for t = 1:m
matrix2 = blkdiag(matrix2,matrix);
end
H = repmat(K, m, m) + matrix2 / gamma; 
% H = repmat(K, m, m) + repmat(matrix,m,m) / gamma; 

P = zeros(m*l, m); 
for t = 1: m
idx1 = l * (t - 1) + 1; 
idx2 = l * t; 

H(idx1: idx2, idx1: idx2) = H(idx1: idx2, idx1: idx2) + K*(m/lambda); 

P(idx1: idx2, t) = ones(l, 1); 
end

eta = H \ P; 
nu = H \ trnY(:); 
S = P'*eta; 
b = inv(S)*eta'*trnY(:); 
alpha = nu - eta*b; 
alpha = reshape(alpha, l, m); 

function w = find_temporal_matrix(n,tau)
w = 1:n;
% w = 1./(w.^tau);
% w = diag(w);
w = diag(w.^tau);
