function predictY = MT_TSVR_tst(tstX, tstY, trnX, alpha, b, lambda, p)

m = size(tstY, 2); 

tstN = size(tstX, 1); 
b = b(:); 
    
K = Kerfun('rbf', tstX, trnX, p, 0); 
predictY = repmat(sum(K*alpha, 2), 1, m) + K*alpha*(m/lambda) + repmat(b', tstN, 1); 
