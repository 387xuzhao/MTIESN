function stateCollectMat = compute_statematrix(inputSequence,trainOutputSequence,...
    nForgetPoints,nInternalUnits,nOutputUnits,internalWeights,inputWeights,outputbackWeights)
%%%% compute_statematrix through the ESN and obtain reservoir states into stateCollectMat.
nDataPoints = length(inputSequence(:,1));
collectIndex = 0;
for i = 1:nDataPoints
    in = inputSequence(i,:)';
    if i==1
        internalState = tansig(inputWeights*in+internalWeights*zeros(nInternalUnits,1)+outputbackWeights*zeros(nOutputUnits,1));
    else
      internalState = tansig(inputWeights*in+internalWeights*totalstate(:,:)+outputbackWeights*trainOutputSequence(i-1,:)');
    end 
  totalstate = internalState;
    if nForgetPoints >= 0 &&  i > nForgetPoints
        collectIndex = collectIndex + 1;
        stateCollectMat(collectIndex,:) = internalState';
    end  
end
