function teachCollectMat = compute_twoteacher(outputSequence1,outputSequence2,nForgetPoints)
% delete the first nForgetPoints elements from outputSequence
if nForgetPoints >= 0
    teachCollectMat = [outputSequence1(nForgetPoints+1:end,:),outputSequence2(nForgetPoints+1:end,:)] ;
end
