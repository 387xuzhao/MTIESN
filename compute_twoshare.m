function stateCollectMat = compute_twoshare(trainInputSequence1,trainInputSequence2,trainOutputSequence1,trainOutputSequence2, ...
        ntrainForgetPoints,nInternalUnits1,nInternalUnits2,nOutputUnits1,nOutputUnits2, ...
      internalWeights1,internalWeights2,inputWeights1,inputWeights2,outputbackWeights1,outputbackWeights2) ...
%%%%% compute_statematrix through the ESN and obtain reservoir states into stateCollectMat.
nDataPoints = length(trainInputSequence1(:,1));
collectIndex = 0;
for i = 1:nDataPoints
    in1 = trainInputSequence1(i,:)';
    in2 = trainInputSequence2(i,:)';
    if i==1
        internalState1 = tansig(inputWeights1*in1+internalWeights1*zeros(nInternalUnits1,1)+outputbackWeights1*zeros(nOutputUnits1,1));
        internalState2 = tansig(inputWeights2*in2+internalWeights2*zeros(nInternalUnits2,1)+outputbackWeights2*zeros(nOutputUnits2,1));
    else
        internalState1 = tansig(inputWeights1*in1+internalWeights1*totalstate1(1:end,:)+outputbackWeights1*trainOutputSequence1(i-1,:)');
        internalState2 = tansig(inputWeights2*in2+internalWeights2*totalstate2(1:end,:)+outputbackWeights2*trainOutputSequence2(i-1,:)');
    end
    totalstate1 = internalState1;
    totalstate2 = internalState2;
    if ntrainForgetPoints >= 0 &&  i > ntrainForgetPoints 
        collectIndex = collectIndex + 1;
        stateCollectMat1(collectIndex,:) =  internalState1';
        stateCollectMat2(collectIndex,:) =  internalState2';
    end  
end
stateCollectMat(:,:) = [stateCollectMat1,stateCollectMat2];
